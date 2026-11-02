# Experiment environment capture

The following environment details were recorded from the same macOS virtual environment used for the reported experiments.

| Component | Captured value |
|---|---|
| Python | 3.12.4 |
| Operating system | macOS 26.5 |
| Architecture | Apple Silicon ARM64 |
| PyTorch | 2.15.0 |
| Apple MPS backend | Available (`True`) |

The full package freeze is preserved in `requirements-macos-mps-lock.txt`.

## Recreating the captured environment

This lock file records the installed packages from one Apple Silicon/macOS experiment environment. It is platform-specific, not a cross-platform dependency specification. Use Python 3.12.4 on Apple Silicon for the closest match.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-macos-mps-lock.txt
```

Verify PyTorch and MPS:

```bash
python -c "import platform, torch; print(platform.platform()); print(torch.__version__); print('MPS:', torch.backends.mps.is_available())"
```

Expected captured values: macOS 26.5 ARM64, PyTorch 2.15.0, MPS `True`.

## Notes

- The lock file is an environment snapshot from the research machine, not a cross-platform dependency specification.
- Re-running experiments on CPU, CUDA, another macOS release, or a different package build may change runtime and small numerical details.
- Keep this snapshot separate from any future generalized dependency file.
