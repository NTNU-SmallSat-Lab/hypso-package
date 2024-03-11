# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Hypso Package'
copyright = 'Norwegian University of Science and Technology (NTNU)'
author = 'Alvaro Flores-Romero'
release = '1.9.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc', # Core library for html generation from docstrings
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',  # Create neat summary tables
    'autoapi.extension',
    'sphinx.ext.githubpages'
]
exclude_patterns = []

autodoc_typehints = 'description'
extensions.append('autoapi.extension')
autoapi_dirs = ['../hypso']
autoapi_template_dir = "_templates/autoapi"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
