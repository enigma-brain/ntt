# Release Process

How new versions of this repository are cut. The package is not published to PyPI: a
release is a git tag plus a GitHub Release, and users install from source (`uv sync`) at
that tag.

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

## Release Steps

### 1. Prepare the release

1. **Update the version** in `pyproject.toml`:
   ```toml
   [project]
   version = "x.y.z"
   ```

2. **Run the checks** that CI runs (`.github/workflows/build_env_run_tests.yml`):
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run pyright
   uv run pytest
   ```

3. **Commit and merge** the version bump to `main` through a pull request. CI must be green.

### 2. Tag and publish

1. **Create and push the tag** from the merged commit on `main`:
   ```bash
   git tag vx.y.z
   git push origin vx.y.z
   ```

2. **Create a GitHub Release** from that tag on the [Releases page](../../releases) and
   describe the changes (see below). There is no automated publishing step: the CI workflow
   runs on pushes and pull requests to `main`, not on tags.

## Release Notes

Include:
- **New features** and improvements
- **Bug fixes**
- **Breaking changes** (if any)
- **Dependency** updates (if significant)

## Downstream submodule

The `downstream/` submodule is pinned to a tag of
[ntt-downstream](https://github.com/enigma-brain/ntt-downstream) (the tag named in
`branch` in `.gitmodules`). When the pin moves, update the submodule, commit the new
gitlink together with the `.gitmodules` tag name, and mention it in the release notes.

## Hotfix Releases

For urgent bug fixes:

1. **Create a hotfix branch** from the release tag:
   ```bash
   git checkout -b hotfix/x.y.z+1 vx.y.z
   ```

2. **Make minimal changes** to fix the issue.
3. **Follow the normal release process** with a patch version bump.
4. **Merge back** to `main` if needed.
