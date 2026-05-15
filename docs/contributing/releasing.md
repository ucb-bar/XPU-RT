# Releasing CompGen to PyPI

Releases are driven by `.github/workflows/release.yml`. Tag a commit, push
the tag, and the workflow builds + publishes to PyPI via Trusted
Publishing (no stored token).

## One-time setup

1. On [pypi.org](https://pypi.org/manage/account/publishing/), add a
   Trusted Publisher for this repo + workflow filename (`release.yml`)
   + environment `pypi`.
2. Repeat on [test.pypi.org](https://test.pypi.org/manage/account/publishing/)
   with environment `testpypi` for dry-run publishes.
3. In the GitHub repo, **Settings → Environments** → create `pypi` and
   `testpypi`. Add required reviewers if you want a human approval
   gate before a real publish.

## Release checklist

1. Bump the version in:
   - `pyproject.toml` (`version = "X.Y.Z"`)
   - `python/compgen/__init__.py` (`__version__`)
2. Refresh the lockfile: `uv lock`
3. Run `make ci` locally — lint + typecheck + tests + lockfile check.
4. Commit: `chore(release): bump version to X.Y.Z`.
5. Tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
6. The `release` workflow runs automatically on the tag push:
   - `build` job packages the wheel + sdist and verifies the tag
     matches the package version.
   - `publish` job uploads to PyPI and creates a GitHub Release.

## Dry run

Before your first real publish, rehearse against TestPyPI:

```bash
gh workflow run release.yml -f dry_run=true
```

The workflow publishes to `https://test.pypi.org/project/compgen/` —
verify install in a throwaway venv:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ compgen==X.Y.Z
```

## Rollback

If a broken release lands on PyPI:

1. **Do not delete** the release from PyPI — PyPI does not allow
   republishing the same version, and yanking is preferred.
2. `Yank` the version from PyPI ("Manage project → Options → Yank").
3. Fix the bug, bump to `X.Y.(Z+1)`, tag, push.

## Troubleshooting

- **Trusted Publishing auth fails**: confirm the repo, workflow filename,
  and environment name exactly match the Publisher config.
- **Tag / version mismatch**: the `build` job fails the check if the
  tag does not equal `compgen.__version__`. Bump both, re-tag.
- **TestPyPI missing dependencies**: TestPyPI doesn't mirror runtime
  deps; use `--extra-index-url https://pypi.org/simple/` when
  installing the dry-run artifact.
