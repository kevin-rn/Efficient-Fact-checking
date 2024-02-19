# Grounding-LM

### Experiment Setup
For our experiments we used the processed [2017 English wikipedia dump](https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2) provided by [HotPotQA](https://hotpotqa.github.io/wiki-readme.html) and using the the original [HoVer work](https://github.com/hover-nlp/hover).

To use a different wikipedia dump, download it from `https://dumps.wikimedia.org/` (for example the [latest one](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2)).  
Afterwards process it using the [forked WikiExtractor](https://github.com/qipeng/wikiextractor) of HotPotQA with the command (Takes a few hours):
```
python -m wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml.bz2 -o extracted --no-templates -c --json

mv -r extracted ../baselines/hover/data/enwiki_files
``` 
Lastly run `python enwiki_processing.py --start_loc=extracted --end_loc=enwiki-latest --process_function=preprocess` for splitting up the text into sentences. This requires [StanfordCoreNLP](https://stanfordnlp.github.io/CoreNLP/) to be set up and running.

Follow below instructions for running our adjustments. Please note that these include placeholder for names in capital letters (e.g. FOLDER_NAME) or between square brackets are either the functions options (e.g. cite, wice) or optional arguments (e.g. --first_paragraph_only) for the scripts.
### Processing English Wikipedia corpus:
* Performing supporting facts detection methods:
```
python enwiki_processing.py --start_loc=FOLDER_NAME --end_loc=FOLDER_NAME_2 --process_function=[claimbuster | wice | cite]

Positional arguments:
--start_loc: The English wikipedia folder to process for.
--end_loc: The target directory to store endresults to.
--process_function: Type of supporting facts functionality to use.

```

### Create Database for HoVer pipeline:
* Generate database file for HoVer pipeline and calculates raw file size as single json file.
```
python database_creation.py --setting=FOLDER_NAME [--split_sent] [--first_para_only] [--pre_compute_embed]

Optional Arguments:
--split_sent: Store sentences per article individually instead of concatenated.
--first_para_only: Store only the lead section of a wikipedia article instead of the entire text.
--pre_compute_embed: Pre-compute vector embeddings for Dense Retrieval.
```
or run the `./cmds/prepare_data_settings.sh ENWIKI_FOLDER_NAME`





## Coding Practices

### Auto-formatting code
1. Install `black`: ```pip install black``` or ```conda install black```
2. In your IDE: Enable formatting on save.
3. Install `isort`: ```pip install isort``` or ```conda install isort```
4. In your IDE: Enable sorting import on save.

In VS Code, you can do this using the following config:
```json
{
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

### Type hints
Use [type hints](https://docs.python.org/3/library/typing.html) for __everything__! No exceptions.

### Docstrings
Write a docstring for __every__ function (except the main function). We use the [Google format](https://github.com/NilsJPWerner/autoDocstring/blob/HEAD/docs/google.md). In VS Code, you can use [autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring).

### Example
```python
def sum(a: float, b: float) -> float:
    """Compute the sum of a and b.

    Args:
        a (float): First number.
        b (float): Second number.
    
    Returns:
        float: The sum of a and b.
    """

    return a + b
```
