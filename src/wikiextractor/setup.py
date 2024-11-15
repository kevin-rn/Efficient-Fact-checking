from setuptools import find_packages, setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="wikiextractor",
    description="A script that extracts and cleans text from a Wikipedia"
    "database dump",
    author="Giuseppe Attardi",
    author_email="attardi@di.unipi.it",
    version="2.69",
    url="https://github.com/attardi/wikiextractor",
    license="GPL 3.0",
    keywords=["text", "nlp"],
    packages=find_packages(),
    py_modules=["WikiExtractor"],
    entry_points={
        "console_scripts": [
            "wikiextractor = WikiExtractor:main",
        ],
    },
)
