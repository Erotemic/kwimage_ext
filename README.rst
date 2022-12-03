The kwimage_ext Module
======================

|ReadTheDocs| |Pypi| |Downloads| 

The ``kwimage_ext`` module, which contains binary extensions for the ``kwimage`` module.

Setting the environment variable ``DISABLE_C_EXTENSIONS=1`` will disable C
extensions at compile time, but this package is mostly C-extensions, so that's
not very useful.


Note, that when building from source, the build may fail if you not in a fresh
state (related to
`skbuild-386 <https://github.com/scikit-build/scikit-build/issues/386>`_. You
can mitigate this by running ``python setup.py clean`` to remove build
artifacts. Building from a clean environment should work.

+------------------+----------------------------------------------+
| Read the docs    | https://kwimage_ext.readthedocs.io           |
+------------------+----------------------------------------------+
| Pypi             | https://pypi.org/project/kwimage_ext         |
+------------------+----------------------------------------------+


.. |CircleCI| image:: https://circleci.com/gh/Erotemic/kwimage_ext.svg?style=svg
    :target: https://circleci.com/gh/Erotemic/kwimage_ext

.. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/github/Erotemic/kwimage_ext?branch=main&svg=True
   :target: https://ci.appveyor.com/project/Erotemic/kwimage_ext/branch/main

.. |Codecov| image:: https://codecov.io/github/Erotemic/kwimage_ext/badge.svg?branch=main&service=github
   :target: https://codecov.io/github/Erotemic/kwimage_ext?branch=main

.. |Pypi| image:: https://img.shields.io/pypi/v/kwimage_ext.svg
   :target: https://pypi.python.org/pypi/kwimage_ext

.. |Downloads| image:: https://img.shields.io/pypi/dm/kwimage_ext.svg
   :target: https://pypistats.org/packages/kwimage_ext

.. |ReadTheDocs| image:: https://readthedocs.org/projects/kwimage_ext/badge/?version=latest
    :target: http://kwimage_ext.readthedocs.io/en/latest/

.. |CodeQuality| image:: https://api.codacy.com/project/badge/Grade/4d815305fc014202ba7dea09c4676343   
    :target: https://www.codacy.com/manual/Erotemic/kwimage_ext?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Erotemic/kwimage_ext&amp;utm_campaign=Badge_Grade

.. |GithubActions| image:: https://github.com/Erotemic/kwimage_ext/actions/workflows/tests.yml/badge.svg?branch=main
    :target: https://github.com/Erotemic/kwimage_ext/actions?query=branch%3Amain
