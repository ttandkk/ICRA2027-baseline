# Realtime VLA Flash environment

Use the project's `uv` workflow so `uv.lock` controls the dependency resolution. The main policy server and LIBERO client are separate environments.

```bash
uv venv --python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv run python -c "import openpi"
```

Install LIBERO client dependencies in a separate environment following `examples/libero/README.md`; its Dockerfile is the supported reproducible runtime. Download checkpoints with Git LFS or the project's documented model commands rather than committing them.
