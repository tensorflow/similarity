from setuptools import setup

setup(
    entry_points={
        "console_scripts": [
            "main = tensorflow_similarity.indexer.cmdline:main_func",
        ]
    }
)