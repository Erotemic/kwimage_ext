"""Release-configuration guards for the Rust-first packaging path."""
from pathlib import Path


REPO_DPATH = Path(__file__).resolve().parents[1]


def test_pyproject_is_authoritative_release_metadata():
    text = (REPO_DPATH / 'pyproject.toml').read_text()
    assert 'requires-python = ">=3.10"' in text
    assert 'license = "Apache-2.0"' in text
    assert 'license = { text =' not in text
    for version in ['3.10', '3.11', '3.12', '3.13', '3.14']:
        assert f'"Programming Language :: Python :: {version}"' in text

    # The legacy build entrypoint must not maintain a second dependency or
    # support-floor declaration. Production metadata belongs to pyproject.toml.
    setup_text = (REPO_DPATH / 'setup.py').read_text()
    main_text = setup_text.split('if __name__ == "__main__":', 1)[1]
    assert 'install_requires' not in main_text
    assert 'extras_require' not in main_text
    assert 'python_requires' not in main_text


def test_ci_installs_exact_wheel_artifact():
    github_text = (REPO_DPATH / '.github/workflows/tests.yml').read_text()
    gitlab_text = (REPO_DPATH / '.gitlab-ci.yml').read_text()

    stale_version_resolver = 'kwimage_ext[$INSTALL_EXTRAS]==$MOD_VERSION'
    assert stale_version_resolver not in github_text
    assert stale_version_resolver not in gitlab_text

    direct_ref = 'kwimage_ext[$INSTALL_EXTRAS] @ $WHEEL_FPATH'
    assert direct_ref in github_text
    assert direct_ref in gitlab_text
    assert '--force-reinstall' in github_text
    assert '--force-reinstall' in gitlab_text

    # GitHub builds Linux/macOS/Windows wheels in parallel. Each test leg should
    # download only its own wheel instead of merging every platform artifact.
    assert 'name: wheels-${{ matrix.os }}-${{ matrix.arch }}' in github_text
    test_job = github_text.split('  test_binpy_wheels:', 1)[1].split('  test_deploy:', 1)[0]
    assert 'pattern: wheels-*' not in test_job


def test_release_version_and_rust_manifest_agree():
    import re

    pyproject_text = (REPO_DPATH / 'pyproject.toml').read_text()
    init_text = (REPO_DPATH / 'kwimage_ext/__init__.py').read_text()
    cargo_text = (REPO_DPATH / 'rust/Cargo.toml').read_text()

    project_block = pyproject_text.split('[project]', 1)[1].split('[project.optional-dependencies]', 1)[0]
    project_version = re.search(r'^version = "([^"]+)"$', project_block, re.MULTILINE).group(1)
    init_version = re.search(r"^__version__ = '([^']+)'$", init_text, re.MULTILINE).group(1)
    cargo_package = cargo_text.split('[package]', 1)[1].split('[lib]', 1)[0]
    cargo_version = re.search(r'^version = "([^"]+)"$', cargo_package, re.MULTILINE).group(1)

    assert project_version == init_version == cargo_version
    assert 'manifest-path = "rust/Cargo.toml"' in pyproject_text
    assert 'features = ["abi3-py310", "extension-module"]' in cargo_text
