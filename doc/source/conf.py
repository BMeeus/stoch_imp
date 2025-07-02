# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Stochastic Impedance'
copyright = '2025, Branko Meeus'
author = 'Branko Meeus'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.coverage']

templates_path = ['_templates']
exclude_patterns = []
highlight_language = 'default'
coverage_show_missing_items = True
autodoc_typehints = "description"
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'classic'
html_static_path = ['_static']

import sys
from pathlib import Path

sys.path.insert(0, str(Path('..', '..').resolve()))