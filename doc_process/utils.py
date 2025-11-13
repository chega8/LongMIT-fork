import jsonlines
import numpy as np
from contextlib import contextmanager

@contextmanager
def read_jsonl_file(file_path: str):
    with open(file_path, 'r') as file:
        with jsonlines.Reader(file) as reader:  # type: ignore
            yield reader


@contextmanager
def memmap(*args, **kwargs):
    reference = np.memmap(*args, **kwargs)
    try:
        yield reference
    finally:
        try:
            reference.flush()
        except Exception:
            pass
        del reference