The kwimage_ext Module
======================

|GitlabCIPipeline| |GitlabCICoverage| |Pypi| |PypiDownloads| |ReadTheDocs|

+------------------+-----------------------------------------------------------+
| Read the docs    | https://kwimage_ext.readthedocs.io                        |
+------------------+-----------------------------------------------------------+
| Gitlab (main)    | https://gitlab.kitware.com/computer-vision/kwimage_ext    |
+------------------+-----------------------------------------------------------+
| Pypi             | https://pypi.org/project/kwimage_ext                      |
+------------------+-----------------------------------------------------------+


The ``kwimage_ext`` module contains acceleration extensions used by
``kwimage``.  Normal installs are Rust-first: the public compatibility API is
implemented in Python and dispatches supported kernels to the
``kwimage_ext._rust`` PyO3 extension.

Install a released wheel in the usual way::

    python -m pip install kwimage_ext

Building from source requires a Rust toolchain because the normal PEP-517
backend is ``maturin``.  CMake and Cython are not required for the production
wheel; they are only needed by maintainers when building the historical parity
backend.


.. |CircleCI| image:: https://circleci.com/gh/Erotemic/kwimage_ext.svg?style=svg
    :target: https://circleci.com/gh/Erotemic/kwimage_ext

.. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/github/Erotemic/kwimage_ext?branch=main&svg=True
   :target: https://ci.appveyor.com/project/Erotemic/kwimage_ext/branch/main

.. |Codecov| image:: https://codecov.io/github/Erotemic/kwimage_ext/badge.svg?branch=main&service=github
   :target: https://codecov.io/github/Erotemic/kwimage_ext?branch=main

.. |Pypi| image:: https://img.shields.io/pypi/v/kwimage_ext.svg
   :target: https://pypi.python.org/pypi/kwimage_ext

.. |PypiDownloads| image:: https://img.shields.io/pypi/dm/kwimage_ext.svg
   :target: https://pypistats.org/packages/kwimage_ext

.. |ReadTheDocs| image:: https://readthedocs.org/projects/kwimage_ext/badge/?version=latest
    :target: http://kwimage_ext.readthedocs.io/en/latest/

.. |CodeQuality| image:: https://api.codacy.com/project/badge/Grade/4d815305fc014202ba7dea09c4676343
    :target: https://www.codacy.com/manual/Erotemic/kwimage_ext?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Erotemic/kwimage_ext&amp;utm_campaign=Badge_Grade

.. |GithubActions| image:: https://github.com/Erotemic/kwimage_ext/actions/workflows/tests.yml/badge.svg?branch=main
    :target: https://github.com/Erotemic/kwimage_ext/actions?query=branch%3Amain

.. |GitlabCIPipeline| image:: https://gitlab.kitware.com/computer-vision/kwimage_ext/badges/main/pipeline.svg
   :target: https://gitlab.kitware.com/computer-vision/kwimage_ext/-/jobs

.. |GitlabCICoverage| image:: https://gitlab.kitware.com/computer-vision/kwimage_ext/badges/main/coverage.svg
    :target: https://gitlab.kitware.com/computer-vision/kwimage_ext/commits/main

Rust-first development
----------------------

The default PEP-517 build uses ``maturin`` and installs the Rust extension as
``kwimage_ext._rust``.  The public Python modules remain compatibility shims,
so downstream ``kwimage`` imports do not need to change while kernels are
ported.

For an editable Rust build::

    ./dev/build_rust.sh
    python -m pytest tests/test_rust_backend.py tests/test_rust_shims.py

The historical Cython/C sources are retained temporarily as a parity reference.
To rebuild both implementations and require direct semantic parity::

    ./dev/check_backend_parity.sh

``dev/build_legacy.sh`` is available when only the reference extensions are
needed.  It builds CPU reference modules under explicit ``*_legacy`` names;
those binaries are not part of the normal wheel.

Backend selection is capability-aware: a Rust module is selected only when it
implements every symbol required by a particular shim.  Set
``KWIMAGE_EXT_FORCE_RUST=1`` to prove that a test is not accidentally falling
through to a legacy extension.  ``KWIMAGE_EXT_FORCE_LEGACY=1`` is intended for
maintainer diagnostics only.

The Rust module also exposes narrow ``coco-assignment`` capabilities through
``kwimage_ext.algo.assignment``. ``coco_greedy_match`` batches IoU thresholds;
``coco_greedy_match_grid`` additionally batches independent area/ignore slices
so downstream evaluators can cross the Python/Rust boundary once per image.
Both accept an already-built CSR sparse candidate graph plus stable prediction
order / truth flags. Geometry and metric policy remain outside these kernels;
downstream kwcoco retains Python reference implementations and fallbacks.
Direct kernel parity and timing can be checked with::

    KWIMAGE_EXT_FORCE_RUST=1 python -m pytest -q tests/test_rust_assignment.py
    python dev/bench_assignment.py --n-pred=400 --n-true=400 --candidates=40

Release wheels use PyO3 ``abi3-py310``.  Build one CPython 3.10 baseline wheel
per platform and test that same artifact on every supported Python version.
The artifact-level check is::

    python dev/check_wheel_artifact.py wheelhouse/kwimage_ext*.whl

Release evidence audit
----------------------

For release candidates, validate the *built wheel*, not an editable checkout.
The artifact validator first checks the archive structure, then installs the
wheel into a temporary target directory and requires every imported
``kwimage_ext`` module (including ``algo.assignment`` and ``_rust``) to resolve
from that isolated target.  This prevents source-tree imports or stale editable
``.pth`` files from hiding an incomplete release artifact::

    python dev/validate_wheel_artifact.py wheelhouse/kwimage_ext*.whl

For a local release candidate, the deterministic preflight is::

    ./dev/build_release_candidate.sh _release_candidate

That command always builds a fresh non-editable wheel, validates the isolated
install/capability surface, runs the release-packaging guards, records SHA256
hashes and wheel contents, and emits an uploadable evidence bundle.
``dev/validate_rust_port.sh`` performs the same non-editable wheel audit in
addition to the editable development tests.  End-to-end COCO metric speed /
memory comparisons remain owned by kwcoco because only the full evaluator can
make a meaningful comparison against ``pycocotools.COCOeval``.
