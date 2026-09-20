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


def test_ci_uses_xcookie_native_reusable_wheel_contract():
    pyproject_text = (REPO_DPATH / 'pyproject.toml').read_text()
    github_text = (REPO_DPATH / '.github/workflows/tests.yml').read_text()
    github_release = (REPO_DPATH / '.github/workflows/release.yml').read_text()
    gitlab_text = (REPO_DPATH / '.gitlab-ci.yml').read_text()

    # Keep the release support range explicit so xcookie does not silently
    # extend CI onto the next prerelease interpreter.
    assert 'max_python = "3.14"' in pyproject_text
    assert 'ci_reusable_wheels = true' in pyproject_text
    assert 'github_url = "https://github.com/Erotemic/kwimage_ext"' in pyproject_text
    assert 'KWIMAGE_EXT_FORCE_RUST = "1"' in pyproject_text
    assert 'python dev/check_wheel_artifact.py wheelhouse/kwimage_ext*.whl' in pyproject_text
    assert './dev/check_backend_parity.sh' in pyproject_text

    # Reusable ABI3 packaging means one build per platform, not one build per
    # Python minor. The same platform artifact is then exercised on 3.10-3.14.
    assert 'build/reusable-linux-x86_64:' in gitlab_text
    assert 'build/cp311-linux-x86_64:' not in gitlab_text
    assert 'cp315' not in gitlab_text
    assert 'cp315' not in github_text
    assert 'Download wheel for this platform' in github_text
    assert 'name: wheels-${{ matrix.os }}-${{ matrix.arch }}' in github_text

    # Artifact tests install a path selected from the downloaded wheelhouse,
    # rather than resolving kwimage_ext by version from an external index.
    for text in [github_text, gitlab_text]:
        assert 'kwimage_ext[$INSTALL_EXTRAS]==$MOD_VERSION' not in text
        assert 'WHEEL_FPATH' in text
        assert 'INSTALL_TARGET="${WHEEL_FPATH}' in text
        assert 'KWIMAGE_EXT_FORCE_RUST' in text
        assert 'check_wheel_artifact.py' in text
        assert 'check_backend_parity.sh' in text

    # GitHub release publication is now a separate xcookie-owned workflow.
    assert 'test_deploy:' not in github_text
    assert 'live_deploy:' not in github_text
    assert 'test_deploy:' in github_release
    assert 'live_deploy:' in github_release
    assert 'https://github.com/Erotemic/kwimage_ext/settings/environments' in github_release
    assert 'owner: Erotemic' in github_release


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
