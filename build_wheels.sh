#!/bin/bash
__doc__="""
SeeAlso:
    pyproject.toml
"""
#pip wheel -w wheelhouse .
# python -m build --wheel -o wheelhouse  #  kwimage_ext: +COMMENT_IF(binpy)
cibuildwheel --config-file pyproject.toml --platform linux --arch x86_64  #  kwimage_ext: +UNCOMMENT_IF(binpy)
