import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+'/InternEmbedding')

import json
import logging
import random
import glob
import math
import time
import torch
import numpy as np
import multiprocessing as mp
from InternEmbedding.embedding.eval.metrics import cosine_similarity
from InternEmbedding.embedding.train.training_embedder import initial_model

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
# mp.set_start_method('spawn')

import yaml
from tqdm import tqdm
import subprocess
from omegaconf import OmegaConf
from doc_process.utils import read_jsonl_file, memmap


def read_embedding_conf(config_path: str):
    conf = OmegaConf.load(config_path)
    return conf.embedder, conf.data


def initial_embedder(args):
    embedder, tokenizer = initial_model(args)
    embedder = embedder.to(args.device)
    if args.embedder_ckpt_path and os.path.exists(args.embedder_ckpt_path):
        logger.info(f"Loading embedder checkpoint from {args.embedder_ckpt_path}")
        embedder.load_state_dict(torch.load(args.embedder_ckpt_path))

    embedder.eval()

    logger.info(f"Embedder initialized: name={args.embedder_name}, device={args.device}, max_length={args.max_length}")
    evaluated_embedder = EvaluatedEmbedder(embedder, tokenizer, args.max_length, args.device)

    return evaluated_embedder


def distribute_initial_embedder(config, device_id):
    device = f'cuda:{device_id}'
    config.device = device

    logger.info(f'Embedder loaded on device {device}')

    return initial_embedder(config)


def generate_faiss_config(embed_path: str, dim: int, dt: np.dtype):
    def read_embeddings(fp):
        fl = os.path.getsize(fp)
        nb = fl // dim // dt.itemsize
        print(nb)
        if fl == dim * dt.itemsize * nb:  # no header
            return ("raw", np.memmap(fp, shape=(nb, dim), dtype=dt, mode="r"))
        else:  # assume npy
            vecs = np.load(fp, mmap_mode="r")
            assert vecs.shape[1] == dim
            assert vecs.dtype == dt
            return ("npy", vecs)

    cfg = {}
    files = []
    size = 0
    for fp in tqdm(glob.glob(f'{embed_path}/*/*/*.npy')):
    # for fn in file_names:
        domain, embed_model, fn = fp.split('/')[-3:]
        assert os.path.exists(fp), f"{fp} is missing"
        ft, xb = read_embeddings(fp)
        files.append(
            {"name": fn, "size": xb.shape[0], "dtype": dt.name, "format": ft, "domain": domain, "embed_model": embed_model}
        )
        size += xb.shape[0]

    cfg["size"] = size
    cfg["root"] = embed_path
    cfg["d"] = dim
    cfg["files"] = files

    logger.info(f"Generated FAISS cfg for root={embed_path}, d={dim}, files={len(files)}, size={size}")
    with open('doc_process/config/faiss/template.yaml', 'w') as fw:
        fw.write(yaml.dump(cfg))


def embedding_documents(args):
    embedder_conf, data_conf, process_id = args

    logger.info(f"[P{process_id}] Initializing embedder for embeddings")
    embedder = distribute_initial_embedder(embedder_conf, process_id)
    encode_batch_size = embedder_conf.encode_batch_size
    embed_dim = embedder_conf.mytryoshka_size

    embedder_name = embedder_conf.embedder_name
    output_dir = data_conf.embed_output_dir
    domain = data_conf.domain
    embed_path = os.path.join(output_dir, domain, embedder_name)
    if not os.path.exists(embed_path):
        os.makedirs(embed_path)
    logger.info(f"[P{process_id}] Output path: {embed_path}")

    doc_files = glob.glob(f'{data_conf.input_dir}/{data_conf.doc_glob}')
    random.seed(process_id)
    random.shuffle(doc_files)
    logger.info(f"[P{process_id}] Found {len(doc_files)} source files matching pattern {data_conf.doc_glob}")

    for doc_file in doc_files:
        doc_name = doc_file.split('/')[-1].split('.')[0]

        doc_embed_file = os.path.join(embed_path, f'{doc_name}.npy')
        doc_batch = []
        doc_ids = []

        if doc_embed_file in glob.glob(f'{embed_path}/*.npy'):
            logger.info(f"[P{process_id}] Skip existing embeddings: {doc_embed_file}")
            continue

        with read_jsonl_file(doc_file) as doc_reader:

            result = subprocess.run(['wc', '-l', doc_file], capture_output=True, text=True)
            num_docs = int(result.stdout.split()[0])
            logger.info(f"[P{process_id}] Start embedding file: {doc_file} (lines={num_docs}) -> {doc_embed_file}")

            with memmap(doc_embed_file, np.float32, 'w+', shape=(num_docs, embed_dim)) as embeds:
                t_start = time.time()
                for di, doc in tqdm(enumerate(doc_reader)):
                    doc_batch.append(doc['content'])
                    doc_ids.append(di)

                    if len(doc_batch) == encode_batch_size or di == num_docs-1:
                        if len(doc_batch) == 0:
                            break

                        b_st = time.time()
                        batch_embed = embedder.batch_encode(doc_batch, batch_size=encode_batch_size)
                        b_et = time.time()
                        embeds[doc_ids] = batch_embed
                        logger.info(f"[P{process_id}] Wrote batch embeddings: count={len(doc_ids)} dim={embed_dim} time_encode={b_et-b_st:.3f}s")

                        doc_batch = []
                        doc_ids = []
                logger.info(f"[P{process_id}] Finished file: {doc_file} in {time.time()-t_start:.3f}s")
                
    # generate_faiss_config(output_dir, embed_dim, np.dtype(np.float32))


def distribute_embedding_documents_in_jsonl(config_path: str, num_process_nodes: int):
    embedder_conf, data_conf = read_embedding_conf(config_path)
    output_dir = data_conf.embed_output_dir
    embed_dim = embedder_conf.mytryoshka_size

    args_list = [(embedder_conf, data_conf, pi) for pi in range(num_process_nodes)]

    with mp.Pool(num_process_nodes) as pool:
        for _ in tqdm(pool.imap(embedding_documents, args_list), total=num_process_nodes):
            pass

    logger.info(f"[P{process_id}] Generating FAISS config from {output_dir}")
    generate_faiss_config(output_dir, embed_dim, np.dtype(np.float32))


def distribute_embed_batch(args):
    pi, docs, batch_size, embedder_conf = args
    logger.info(f"[P{pi}] Distributing embed batch of size {len(docs)}")
    embedder = distribute_initial_embedder(embedder_conf, pi)
    batch_embed = embedder.batch_encode(docs, batch_size=batch_size)
    return batch_embed


def distribute_embedding_documents(config_path: str, num_process_nodes: int):
    embedder_conf, data_conf = read_embedding_conf(config_path)

    logger.info("Preparing distributed embedding over documents")
    doc_files = glob.glob(f'{data_conf.input_dir}/{data_conf.doc_glob}')
    encode_batch_size = embedder_conf.encode_batch_size
    embed_dim = embedder_conf.mytryoshka_size

    embedder_name = embedder_conf.embedder_name
    output_dir = data_conf.embed_output_dir
    domain = data_conf.domain
    embed_path = os.path.join(output_dir, domain, embedder_name)
    if not os.path.exists(embed_path):
        os.makedirs(embed_path)
    logger.info(f"Embedding output directory: {embed_path}")

    doc_files = glob.glob(f'{data_conf.input_dir}/{data_conf.doc_glob}')
    for doc_file in doc_files:
        doc_name = doc_file.split('/')[-1].split('.')[0]

        doc_embed_file = os.path.join(embed_path, f'{doc_name}.npy')
        doc_batch = []
        doc_ids = []
        with read_jsonl_file(doc_file) as doc_reader:

            result = subprocess.run(['wc', '-l', doc_file], capture_output=True, text=True)
            num_docs = int(result.stdout.split()[0])
            logger.info(f"Processing file: {doc_file} (lines={num_docs})")
            process_batch_size = encode_batch_size * 100
            global_batch_size = process_batch_size * num_process_nodes

            if num_docs < global_batch_size:
                process_batch_size = math.ceil(num_docs / num_process_nodes)
                global_batch_size = num_docs

            with memmap(doc_embed_file, np.float32, 'w+', shape=(num_docs, embed_dim)) as embeds:
                for di, doc in tqdm(enumerate(doc_reader)):
                    doc_batch.append(doc['content'])
                    doc_ids.append(di)

                    if len(doc_batch) == global_batch_size or di == num_docs-1:
                        if len(doc_batch) == 0:
                            break

                        args_list = []
                        global_batch_ids = []
                        for pi, i in enumerate(range(0, len(doc_batch), process_batch_size)):
                            process_docs = doc_batch[i:i+process_batch_size]
                            process_doc_ids = doc_ids[i:i+process_batch_size]
                            global_batch_ids.append(process_doc_ids)
                            args_list.append((pi, process_docs, encode_batch_size, embedder_conf))

                        with mp.Pool(processes=len(args_list)) as pool:
                            global_batch_embeds = []
                            for e in tqdm(pool.imap(distribute_embed_batch, args_list), total=len(args_list)):
                                global_batch_embeds.append(e)

                            for ids, es in zip(global_batch_ids, global_batch_embeds):
                                embeds[ids] = es

                        doc_batch = []
                        doc_ids = []

    logger.info(f"Generating FAISS config from {output_dir}")
    generate_faiss_config(output_dir, embed_dim, np.dtype(np.float32))



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='embed_docs')
    parser.add_argument('--config', type=str, default='doc_process/config/embedding/embedding_example.yaml', help='embedding config')
    parser.add_argument('--num_process_nodes', type=int, default=8, help='Number of GPUs')
    parser.add_argument('--precomputed_embed_root', type=str, default='', help='If set, skip embedding and only generate FAISS cfg from this root')
    args = parser.parse_args()

    # set seeds per process for reproducibility
    try:
        import numpy as np
    except Exception:
        np = None
    try:
        import torch
    except Exception:
        torch = None

    base_seed = 0
    random.seed(base_seed)
    if np is not None:
        try:
            np.random.seed(base_seed)
        except Exception:
            pass
    if torch is not None:
        try:
            torch.manual_seed(base_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(base_seed)
        except Exception:
            pass

    # If user supplies precomputed embeddings, only generate FAISS cfg and exit
    if isinstance(args.precomputed_embed_root, str) and len(args.precomputed_embed_root) > 0:
        embedder_conf, data_conf = read_embedding_conf(args.config)
        logger.info(f"Using precomputed embeddings at {args.precomputed_embed_root}, generating FAISS config only")
        generate_faiss_config(args.precomputed_embed_root, embedder_conf.mytryoshka_size, np.dtype(np.float32))
    else:
        # for documents
        distribute_embedding_documents(args.config, args.num_process_nodes)