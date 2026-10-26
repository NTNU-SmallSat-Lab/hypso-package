"""Generate demo_processing.ipynb from demo_processing.py (cells split at
`# %%`, `# %% [markdown]` cells as markdown). Run after editing the script:

    python make_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def cells(source: str) -> list:
    chunks = []          # (header, lines): a cell starts at a line beginning with "# %%"
    for line in source.splitlines():
        if line.startswith("# %%"):
            chunks.append((line[4:].strip(), []))
        elif chunks:
            chunks[-1][1].append(line)
    out = []
    for header, lines in chunks:
        body = "\n".join(lines).strip("\n")
        if header == "[markdown]":
            text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
            out.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
        else:
            out.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                        "source": body.splitlines(keepends=True)})
    return out


notebook = {"cells": cells((HERE / "demo_processing.py").read_text()),
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}
(HERE / "demo_processing.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
print(f"{len(notebook['cells'])} cells")
