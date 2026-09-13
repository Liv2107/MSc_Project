# Tests

Run `python -m pytest` from the repository root. Tests use temporary files,
synthetic images and mock backbones; no research dataset, downloaded weights or
GPU is needed.

The suite covers manifest validation, group and test-set isolation, metrics,
model shapes, freezing, checkpoint round trips, resuming, experiment protocols,
aggregation and plotting. Small integration tests exercise training end to end.

For a focused check, pass a file, for example:

```sh
python -m pytest tests/test_resume_contracts.py
```
