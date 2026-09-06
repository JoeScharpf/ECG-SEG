# Paper benchmark — boundary-aware Mean Teacher (ResNet-18 + U-Net)

Test MeanIoU, mean ± std over seeds 0/1/2.

| Dataset | 1/16 | 1/8 | 1/4 | 1/2 |
|---------|-----:|----:|----:|----:|
| LUDB | 0.8454±0.0005 | 0.8552±0.0009 | 0.8545±0.0025 | 0.8591±0.0016 |
| QTDB | 0.6228±0.0302 | 0.6772±0.0049 | 0.7511±0.0011 | 0.7683±0.0057 |
| ISP | 0.7669±0.0019 | 0.7790±0.0019 | 0.7933±0.0022 | 0.7992±0.0006 |
| Zhejiang | 0.8372±0.0006 | 0.8478±0.0007 | 0.8536±0.0001 | 0.8616±0.0008 |

Cross-domain (merged): 0.8261±0.0011

Regenerate:

```bash
python baseline/aggregate_benchmark_table.py
```
