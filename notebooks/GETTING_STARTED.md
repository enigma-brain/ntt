# Notebooks

Jupyter notebooks that show how to use the package.

## Contents

- **`v0.ipynb`**: End-to-end walkthrough of building a tokenizer: defining a `TokenizerSpec`, implementing the `BaseTokenizer`, registering it, wiring up `NTTDataModule` and running a toy training loop.
- **`example_usage.ipynb`**: Shorter usage snippets.
- **`compose.ipynb`**: Composing Hydra configs from Python.

## Usage

After installing the project's Python environment with `uv`, open the notebooks in VSCode or Jupyter and select the project's Python interpreter as the kernel.

## Guidelines

- Remove outputs before committing (or use nbstripout).
- Move reusable code into the main package.
