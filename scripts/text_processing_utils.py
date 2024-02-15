import bz2
import cchardet # speed up lxml (html parsing) just by importing
import contextlib
import json
import lxml
import os
import warnings

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning 
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Any

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument
    ref: https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution

    Parameters:
        tqdm_object (tqdm): tqdm object for multiprocessing.
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def search_file_paths(dir: str, suffix: str = ".bz2") -> List[str]:
    """
    Retrieves all bz2 file paths within a specified directory.

    Parameters:
        dir (str): directory to search through.

    Returns:
        List of bz2 file path strings e.g. ['/AA/wiki_00.bz2', ..]
    """
    file_paths = []
    for subdir, _, files in os.walk(dir):
        for file in files:
            bz2_filepath = os.path.join(subdir, file)
            if bz2_filepath.endswith(suffix):
                file_paths.append(bz2_filepath[len(dir) :])
    return file_paths

def save_file_to_path(json_list: List[Any], dir: str, filepath: str) -> None:
    """
    Writes json objects to a bz2 file for a given filepath. 
    """
    folderpath = dir + os.path.split(filepath)[0]
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    with bz2.BZ2File(dir + filepath, "wb") as bz2_f:
        for j_obj in json_list:
            json_data = json.dumps(j_obj)
            bz2_f.write(json_data.encode("utf-8"))
            bz2_f.write(b"\n")

def multiprocess_bz2(func: Any, start_location: str, end_location: str, n_processes: int=16, process_style=None) -> Any:
    """
    Performs multiprocessing for a given function and its filepaths.
    """
    # Get all filepaths to still process for.
    file_paths = search_file_paths(start_location)
    exclude_paths = search_file_paths(end_location)
    search_paths = list(set(file_paths).symmetric_difference(set(exclude_paths)))
    print(f"total files: {len(file_paths)}, pending: {len(search_paths)}")

    # Start Multiprocessing using joblib.
    with tqdm_joblib(tqdm(desc="Process bz2 file", total=len(search_paths))) as progress_bar:
        results = Parallel(n_jobs=n_processes, prefer=process_style)(
            delayed(func)(bz2_filepath, start_location, end_location) for bz2_filepath in search_paths
        )

    return results

def remove_html_tags(sentences: List[str]) -> List[str]:
    """
    Removes html tags from string.

    Parameters:
        - sentences (List[str]): list of sentences possibly containing html tags.
    """
    result = []
    for sent in sentences:
        soup = BeautifulSoup(sent, features="lxml")
        result.append(soup.get_text(strip=False))
    return result

def get_file_iter(file: Any, filepath: str) -> tqdm:
    """
    Get progressbar for bz2 file.
    """
    file_size = sum(1 for _ in file) # total amount of wiki articles
    file.seek(0) # reset read pointer
    return tqdm(file, desc=f"Processing {filepath}", leave=False, total=file_size)