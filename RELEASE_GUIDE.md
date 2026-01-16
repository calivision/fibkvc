# Release Guide for fibkvc

## Single-Source Version Management

The version is now managed in **one file only**: `opensource/VERSION`

All other files automatically read from this source:
- `setup.py` - reads VERSION file
- `pyproject.toml` - uses version from setup.py
- `fibkvc/__init__.py` - reads VERSION file at runtime

## How to Release a New Version

### Step 1: Update the Version

Edit `opensource/VERSION` with the new version number:

```bash
echo "0.1.2" > opensource/VERSION
```

### Step 2: Commit and Tag

```bash
git add opensource/VERSION
git commit -m "Bump version to 0.1.2"
git tag -a v0.1.2 -m "Release version 0.1.2"
git push origin main
git push origin v0.1.2
```

### Step 3: Automated Release

Once you push the tag, GitHub Actions automatically:

1. **Build** - Creates wheels and source distributions for:
   - Python 3.8, 3.9, 3.10, 3.11
   - Ubuntu, Windows, macOS

2. **Create GitHub Release** - Attaches all built artifacts for download

3. **Publish to PyPI** - Uploads to PyPI (requires `PYPI_API_TOKEN` secret configured)

## GitHub Releases & Downloads

Users can download pre-built wheels from:
```
https://github.com/calivision/fibkvc/releases/tag/v0.1.2
```

Each release includes:
- `.whl` files (wheels) for all Python versions and platforms
- `.tar.gz` source distribution

## Local Testing Before Release

To test locally before pushing:

```bash
cd opensource

# Update VERSION
echo "0.1.2" > VERSION

# Build locally
python -m build

# Test installation
pip install dist/fibkvc-0.1.2-py3-none-any.whl

# Verify version
python -c "import fibkvc; print(fibkvc.__version__)"
```

## Setting Up PyPI Token (One-time)

To enable automatic PyPI publishing:

1. Generate a token at https://pypi.org/manage/account/tokens/
2. Add it to GitHub Secrets:
   - Go to repo Settings → Secrets and variables → Actions
   - Create new secret: `PYPI_API_TOKEN`
   - Paste your PyPI token

## Troubleshooting

**Version not updating?**
- Ensure `VERSION` file has no extra whitespace
- Clear build artifacts: `rm -rf build dist *.egg-info`
- Rebuild: `python -m build`

**Release not publishing to PyPI?**
- Check that `PYPI_API_TOKEN` is set in GitHub Secrets
- Verify the token has upload permissions
- Check GitHub Actions logs for errors

**Downloads not appearing on GitHub?**
- Ensure the tag was pushed: `git push origin v0.1.2`
- Check GitHub Actions workflow completed successfully
- Artifacts are automatically attached to the release

## Quick Reference

```bash
# One-liner to bump version and release
VERSION="0.1.2"
echo $VERSION > opensource/VERSION
git add opensource/VERSION
git commit -m "Bump version to $VERSION"
git tag -a v$VERSION -m "Release version $VERSION"
git push origin main
git push origin v$VERSION
```
