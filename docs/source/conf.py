# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os

import sys
import textwrap


def get_project_version():
    file = open("../../version.txt")
    version = file.read()
    file.close()
    return version


project = 'EvSpikeSim'
copyright = '2023, Florian Bacho'
author = 'Florian Bacho'
release = get_project_version()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx_copybutton',
              'sphinx.ext.autodoc',
              "sphinx.ext.autosummary",
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx.ext.todo',
              'breathe',
              'exhale']

todo_include_todos = True # Shows todo

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

# -- Options for C++ API Documentation ---------------------------------------

# Specify Doxygen XML to Breathe to generate C++ API documentation from source code
breathe_projects = {"EvSpikeSim": "./_doxygen/xml"}
breathe_default_project = "EvSpikeSim"

# Declare decorators
cpp_id_attributes = ["INLINE", "DEVICE", "GLOBAL", "CALLBACK", "__global__", "__device__", "__inline__"]

# Exhale arguments
exhale_args = {
    # These arguments are required
    "containmentFolder": "./cpp_api",
    "rootFileName": "library_root.rst",
    "afterTitleDescription": textwrap.dedent('''
       The following documentation presents the C++ API of EvSpikeSim.
       
       .. note::
       
        The file hierarchy available bellow is representative of the source hierarchy and does not respect the final
        structure of headers in the installed library. Moreover, some features described bellow may be in use in the GPU 
        version and not available in the CPU version (and inversely).
       
    '''),
    "doxygenStripFromPath": "..",
    # Heavily encouraged optional argument (see docs)
    "rootFileTitle": "C++ API",
    # Suggested optional arguments
    "createTreeView": True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin": """
                            INPUT = ../../core/common/inc ../../core/cpu/inc ../../core/gpu/inc
                            ENABLE_PREPROCESSING = YES
                            MACRO_EXPANSION = YES
                            EXPAND_ONLY_PREDEF = YES
                            PREDEFINED += protected=private # Hides protected members
                          """
}