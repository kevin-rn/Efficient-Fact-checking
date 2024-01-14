from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import bz2
import contextlib
import json
import joblib
from joblib import Parallel, delayed
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import warnings
import os
os.environ['PYTHONWARNINGS']='ignore'

data_location= "data/enwiki-20171001-pages-meta-current-withlinks-processed/"
processed_location = "data/enwiki-2017/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
claim_tokenizer = AutoTokenizer.from_pretrained("Nithiwat/bert-base_claimbuster")
claim_model =  AutoModelForSequenceClassification.from_pretrained("Nithiwat/bert-base_claimbuster").to(device)

def search_file_paths(dir):
    file_paths = []
    for subdir, _, files in os.walk(dir):
        for file in files:
            bz2_filepath = subdir + os.sep + file
            if bz2_filepath.endswith('.bz2'):
                file_paths.append(bz2_filepath[len(dir):])
    return file_paths

def remove_html_tags(sentences):
    """
    Removes html tags from string.
    """
    soup = BeautifulSoup(sentences, features='html.parser')
    return soup.get_text()

def extract_claimworthy(sentences):
    """
    Performs claim detection on list of strings and returns the salient ones.
    """
    raw_sentences = [remove_html_tags(sent) for sent in sentences]
    tokenized_inputs = claim_tokenizer(
        raw_sentences, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = claim_model(**tokenized_inputs).logits
        logits = logits.cpu()
    label_indices = torch.nonzero(logits.argmax(dim=1) == 1).squeeze().cpu()
    # Prevent looping over 0d-tensor error.
    if label_indices.dim() == 0:
        label_indices = label_indices.unsqueeze(0)
    claimworthy = [raw_sentences[idx] for idx in label_indices]

    return claimworthy

def read_and_claim_detect_bz2(bz2_path):
    """
    Read bz2 file and add fact_text field containing claim detected sentences
    """
    json_obj = []
    with bz2.open(bz2_path, "rb") as file:
        for line in file:
            wiki_article = json.loads(line)
            article_text = wiki_article['text']

            fact_text = [article_text[0]] # skip title

            # Loop through paragraphs
            for i in range(1, len(article_text)):
                paragraph_sents = article_text[i]
                if paragraph_sents:
                    fact_sents = extract_claimworthy(paragraph_sents)
                    fact_text.append(fact_sents) # may append empty paragraph array (no claims)
            wiki_article['fact_text'] = fact_text
            json_obj.append(wiki_article)
    return json_obj

def write_bz2(bz2_path, json_vals):
    """
    Overwrite wiki articles in bz2
    """
    with bz2.BZ2File(bz2_path, "wb") as bz2_f:
        for j_obj in json_vals:
            json_data = json.dumps(j_obj)
            bz2_f.write(json_data.encode('utf-8'))
            bz2_f.write(b"\n")

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument
    https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
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

def process_file(filepath):
    json_obj = read_and_claim_detect_bz2(bz2_path = data_location + filepath)
    write_bz2(bz2_path=processed_location + filepath, json_vals=json_obj)

def main():
    # discover all bz2 file paths in directory and its descendants
    file_paths = search_file_paths(data_location)
    # Exclude bz2 file paths that have already been processed
    exclude_paths = search_file_paths(processed_location)
    search_paths = list(set(file_paths).symmetric_difference(set(exclude_paths)))
    print(f"total files: {len(file_paths)}, pending: {len(search_paths)}")

    # num_jobs = os.cpu_count() / 2
    num_jobs = 12
    with tqdm_joblib(tqdm(desc="Process bz2 file", total=len(search_paths))) as progress_bar:
        Parallel(n_jobs=num_jobs)(delayed(process_file)(bz2_filepath) for bz2_filepath in search_paths)


if __name__ == "__main__":
    main()