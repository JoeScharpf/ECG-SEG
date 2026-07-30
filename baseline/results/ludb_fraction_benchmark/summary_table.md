# LUDB label-fraction benchmark

Mean ± std test MeanIoU over seeds 0/1/2.

| Method | 1/16 | 1/8 | 1/4 | 1/2 |
|--------|-----:|----:|----:|----:|
| U-Net supervised | 0.8178±0.0024 | 0.8402±0.0010 | 0.8478±0.0006 | 0.8548±0.0010 |
| U-Net + Mean Teacher | 0.8397±0.0014 | 0.8524±0.0009 | 0.8521±0.0014 | 0.8560±0.0003 |
| U-Net + Boundary-aware MT | 0.8454±0.0005 | 0.8552±0.0009 | 0.8545±0.0025 | 0.8591±0.0016 |

\* incomplete seeds. Regenerate with:

```bash
python baseline/aggregate_ludb_fraction_table.py
```
