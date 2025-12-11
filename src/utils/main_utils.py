import math
import os
import sys

import numpy as np
import dill  # type: ignore
import yaml
import nltk
import tiktoken
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging


def num_tokens_from_string(text: str, model_name: str = "cl100k_base") -> int:
    """
    Returns the number of tokens in a text string.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to a common encoding if model_name is not directly supported
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens

def download_nltk_package_if_needed(resource_name, path_prefix):
    """Checks for a specific NLTK resource using the correct path prefix."""
    try:
        # Check the specific expected location
        nltk.data.find(f'{path_prefix}/{resource_name}')
        print(f"'{resource_name}' found in {path_prefix}. Skipping download.")
    except LookupError:
        print(f"'{resource_name}' not found. Downloading...")
        nltk.download(resource_name, quiet=True)
        print("Download complete.")

def is_model_present(model_path) -> bool:
    """Function that checks any model exists or not"""
    try:
        if os.path.exists(model_path):
            return True
    except MyException as e:
        print(e)
        return False

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise MyException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise MyException(e, sys) from e


def compute_k(*, total: int, pct: float | None, upper_bound: int) -> int:
    """
    Convert a percentage to an integer k, clamped to available docs.

    - Uses ceil to avoid losing small fractions.
    - Ensures the value is >= 0 and <= upper_bound.
    """
    if total <= 0 or upper_bound <= 0:
        return 0

    if pct is None:
        return 0

    calculated = int(math.ceil(total * pct))
    return max(0, min(calculated, upper_bound))


def count_documents(vector_store) -> int:
    """
    Safely count documents in a FAISS store.
    """
    ids = getattr(vector_store, "index_to_docstore_id", None)
    if ids is not None:
        try:
            return len(ids)
        except Exception:
            pass

    docstore = getattr(vector_store, "docstore", None)
    if docstore is not None and hasattr(docstore, "_dict"):
        try:
            return len(docstore._dict)
        except Exception:
            pass
    return 0


def load_object(file_path: str) -> object:
    """
    Returns model/object from project directory.
    file_path: str location of file to load
    return: Model/Obj
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        return obj
    except Exception as e:
        raise MyException(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise MyException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise MyException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise MyException(e, sys) from e

