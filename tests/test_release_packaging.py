"""Release-configuration guards for the Rust-first packaging path."""
from pathlib import Path


REPO_DPATH = Path(__file__).resolve().parents[1]


def test_pyproject_is_authoritative_release_metadata():
    text = (REPO_DPATH / 'pyproject.toml').read_text()
    assert 'requires-python = ">=3.10"' in text
    assert 'license = "Apache-2.0"' in text
    assert 'license = { text =' not in text
    for version in ['3.10', '3.11', '3.12', '3.13', '3.14', '3.15']:
        assert f'"Programming Language :: Python :: {version}"' in text

    # The legacy build entrypoint must not maintain a second dependency or
    # support-floor declaration. Production metadata belongs to pyproject.toml.
    setup_text = (REPO_DPATH / 'setup.py').read_text()
    main_text = setup_text.split('if __name__ == "__main__":', 1)[1]
    assert 'install_requires' not in main_text
    assert 'extras_require' not in main_text
    assert 'python_requires' not in main_text


def test_ci_uses_xcookie_native_reusable_wheel_contract():
    pyproject_text = (REPO_DPATH / 'pyproject.toml').read_text()
    github_text = (REPO_DPATH / '.github/workflows/tests.yml').read_text()
    github_checks = (REPO_DPATH / '.github/workflows/checks.yml').read_text()
    github_release = (REPO_DPATH / '.github/workflows/release.yml').read_text()
    gitlab_root = (REPO_DPATH / '.gitlab-ci.yml').read_text()
    gitlab_text = (REPO_DPATH / '.gitlab/ci/main.yml').read_text()
    gitlab_checks = (REPO_DPATH / '.gitlab/ci/checks.yml').read_text()

    # Let xcookie infer the current supported CPython range from the 3.10
    # floor.  This picks up 3.15 as a prerelease lane without another manual
    # ceiling/list edit, while xcookie's prerelease policy keeps it nonblocking.
    assert 'max_python =' not in pyproject_text
    assert 'ci_cpython_versions =' not in pyproject_text
    assert 'ci_reusable_wheels = true' in pyproject_text
    assert 'archs = ["auto64"]' in pyproject_text
    assert 'github_url = "https://github.com/Erotemic/kwimage_ext"' in pyproject_text
    assert 'KWIMAGE_EXT_FORCE_RUST = "1"' in pyproject_text
    assert 'python dev/validate_wheel_artifact.py wheelhouse/kwimage_ext*.whl' in pyproject_text
    assert "torch>=1.11.0; python_version < '3.15'" in pyproject_text
    assert './dev/check_backend_parity.sh' in pyproject_text
    assert "python -m pip install -e '.[tests]'" in pyproject_text

    # Reusable ABI3 packaging means one build per platform, not one build per
    # Python minor. The same platform artifact is then exercised on 3.10-3.15.
    assert 'build/reusable-linux-x86_64:' in gitlab_text
    assert 'build/cp311-linux-x86_64:' not in gitlab_text
    assert 'build/cp315-' not in gitlab_text
    assert 'build = "cp310-*"' in pyproject_text
    assert "python-version: '3.15'" in github_text
    assert "allow-prereleases: 'true'" in github_text
    assert 'image: python:3.15-rc' in gitlab_text
    assert 'Download wheel for this platform' in github_text
    assert 'name: wheels-${{ matrix.os }}-${{ matrix.arch }}' in github_text

    # Artifact tests install a path selected from the downloaded wheelhouse,
    # rather than resolving kwimage_ext by version from an external index.
    for text in [github_text, gitlab_text]:
        assert 'kwimage_ext[$INSTALL_EXTRAS]==$MOD_VERSION' not in text
        assert 'WHEEL_FPATH' in text
        assert 'INSTALL_TARGET="${WHEEL_FPATH}' in text
        assert 'KWIMAGE_EXT_FORCE_RUST' in text
        assert 'validate_wheel_artifact.py' in text

    # Project-owned source checks live in provider-specific generated check
    # files. They must not be duplicated into the normal test or release jobs.
    assert 'check_backend_parity.sh' not in github_text
    assert 'check_backend_parity.sh' not in github_release
    assert 'check_backend_parity.sh' in github_checks
    assert "python -m pip install -e '.[tests]'" in github_checks
    assert 'pull_request:' in github_checks

    assert 'include:' in gitlab_root
    assert '.gitlab/ci/main.yml' in gitlab_root
    assert '.gitlab/ci/checks.yml' in gitlab_root
    assert 'check/backend-parity' not in gitlab_text
    assert 'check/backend-parity' in gitlab_checks
    assert "python -m pip install -e '.[tests]'" in gitlab_checks
    assert './dev/check_backend_parity.sh' in gitlab_checks

    # GitHub release publication is now a separate xcookie-owned workflow.
    assert 'test_deploy:' not in github_text
    assert 'live_deploy:' not in github_text
    assert 'test_deploy:' in github_release
    assert 'live_deploy:' in github_release
    assert 'https://github.com/Erotemic/kwimage_ext/settings/environments' in github_release
    assert 'owner: Erotemic' in github_release

    # Trusted Publishing is explicitly enabled for both mirrors. GitHub keeps
    # its existing dedicated release workflow; GitLab isolates OIDC to a
    # minimal publisher job instead of exposing the token to release plumbing.
    assert 'ci_pypi_trusted_publishing = ["github", "gitlab"]' in pyproject_text
    assert 'GitLab PyPI Trusted Publishing setup checklist' in gitlab_root
    assert 'GitLab instance: https://gitlab.kitware.com' in gitlab_root
    assert 'namespace: computer-vision' in gitlab_root
    assert 'repository: kwimage_ext' in gitlab_root
    assert 'top-level pipeline: .gitlab-ci.yml' in gitlab_root
    assert 'publish/pypi:' in gitlab_text
    publish_text = gitlab_text.split('publish/pypi:', 1)[1]
    assert 'PYPI_ID_TOKEN:' in publish_text
    assert 'aud: pypi' in publish_text
    assert 'name: pypi' in publish_text
    assert 'job: gpgsign/wheels' in publish_text
    assert 'twine upload --skip-existing "$WHEEL_PATH"' in publish_text
    deploy_text = gitlab_text.split('deploy/wheels:', 1)[1].split('publish/pypi:', 1)[0]
    assert 'PYPI_ID_TOKEN' not in deploy_text
    assert 'twine upload' not in deploy_text


def test_release_version_and_rust_manifest_agree():
    import re

    pyproject_text = (REPO_DPATH / 'pyproject.toml').read_text()
    init_text = (REPO_DPATH / 'kwimage_ext/__init__.py').read_text()
    cargo_text = (REPO_DPATH / 'rust/Cargo.toml').read_text()
    cargo_lock_text = (REPO_DPATH / 'rust/Cargo.lock').read_text()

    project_block = pyproject_text.split('[project]', 1)[1].split('[project.optional-dependencies]', 1)[0]
    project_version = re.search(r'^version = "([^"]+)"$', project_block, re.MULTILINE).group(1)
    init_version = re.search(r"^__version__ = '([^']+)'$", init_text, re.MULTILINE).group(1)
    cargo_package = cargo_text.split('[package]', 1)[1].split('[lib]', 1)[0]
    cargo_version = re.search(r'^version = "([^"]+)"$', cargo_package, re.MULTILINE).group(1)

    assert project_version == init_version == cargo_version
    lock_package = cargo_lock_text.split('name = "kwimage_ext"', 1)[1].split('[[package]]', 1)[0]
    lock_version = re.search(r'^version = "([^"]+)"$', lock_package, re.MULTILINE).group(1)
    assert lock_version == project_version
    assert 'manifest-path = "rust/Cargo.toml"' in pyproject_text
    assert 'python-packages = ["kwimage_ext"]' in pyproject_text
    assert 'features = ["abi3-py310", "extension-module"]' in cargo_text



def test_release_validation_is_artifact_isolated():
    validator = (REPO_DPATH / 'dev/validate_wheel_artifact.py').read_text()
    audit = (REPO_DPATH / 'dev/check_release_install.py').read_text()
    build_script = (REPO_DPATH / 'build_wheels.sh').read_text()
    candidate_script = (REPO_DPATH / 'dev/build_release_candidate.sh').read_text()

    assert "'--target'" in validator
    assert "'--require-install-root'" in validator
    assert "PYTHONNOUSERSITE" in validator
    assert 'path.is_relative_to(root)' in audit
    assert 'artifact_isolated' in audit
    assert 'rm -rf wheelhouse' in build_script
    assert 'CIBW_BUILD = <from pyproject.toml>' in build_script
    assert 'LOCAL_CP_VERSION=' not in build_script
    assert 'validate_wheel_artifact.py' in build_script
    assert 'pip wheel --no-deps' in candidate_script
    assert 'validate_wheel_artifact.py' in candidate_script



def test_typecheck_policy_matches_rust_first_backend():
    pyproject_text = (REPO_DPATH / 'pyproject.toml').read_text()
    assignment_text = (REPO_DPATH / 'kwimage_ext/algo/assignment.py').read_text()
    algo_nms_text = (REPO_DPATH / 'kwimage_ext/algo/algo_nms.py').read_text()

    # The compiled PyO3 module is discovered dynamically at runtime, just like
    # the other capability-aware Rust dispatchers. Static checking must not
    # require a generated extension stub merely to import this frontend.
    assert "importlib.import_module('kwimage_ext._rust')" in assignment_text
    assert 'from kwimage_ext import _rust' not in assignment_text

    # GPU NMS is an explicit compatibility stub in the Rust rewrite; the
    # frontend should raise its NotImplementedError without requiring torch.
    assert 'import torch' not in algo_nms_text

    # Keep ty strict over the active package while narrowly relaxing only the
    # pre-existing experimental torch_nms helper, which declares itself broken.
    assert 'include = ["kwimage_ext/algo/_nms_backend/torch_nms.py"]' in pyproject_text
    assert 'unresolved-import = "ignore"' in pyproject_text
    assert 'unresolved-attribute = "ignore"' in pyproject_text
