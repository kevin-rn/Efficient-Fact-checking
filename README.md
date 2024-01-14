# Grounding-LM

Follow below instructions for running our adjustments. Please note that these include placeholder for names in capital letters (e.g. FOLDER_NAME) or between square brackets are either the functions options (e.g. cite, wice) or optional arguments (e.g. --first_paragraph_only) for the scripts.
### Processing English Wikipedia corpus:
* Performing claim extraction methods:
```
python enwiki_processing.py --start_location=FOLDER_NAME --end_location=FOLDER_NAME_2 --process_function=[claimbuster | claimbuster_current | wice | cite] [--first_paragraph_only]  
```
* Generating embeddings for the claim extracted version or original:
```
python enwiki_processing.py --start_location=FOLDER_NAME_2 --end_location=FOLDER_NAME_3 --process_function=[generate_embed | generate_embed_original] [--first_paragraph_only]
```
* Print raw size of the corpus (just the wikipedia article title and text)
```
python enwiki_processing.py --start_location=FOLDER_NAME --end_location=FOLDER_NAME_2 --process_function=print_raw_size
```

### Create Database for HoVer pipeline:
* Generate database file for HoVer pipeline
```
python database_creation.py --setting=FOLDER_NAME_ --db_function=[] [--first_paragraph_only] [--include_embed]
```





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
