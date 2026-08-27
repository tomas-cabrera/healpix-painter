# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "healpix-painter"
copyright = "2026, Tomás Cabrera"
author = "Tomás Cabrera"
release = "0.3.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autosummary_generate_overwrite = True  # Overwrite existing files

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "navbar_align": "left",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/yourorg/yourproject",
            "icon": "fa-brands fa-github",
        },
    ],
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "header_links_before_dropdown": 6,
}

html_sidebars = {
    "api": ["sidebar-nav-bs"],  # left sidebar shows the API nav tree
}

html_context = {
    "default_mode": "auto",
}
