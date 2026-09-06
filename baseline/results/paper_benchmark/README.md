# Paper benchmark tables

Full ResNet-18 + U-Net + boundary-MT benchmark (in-domain + cross-domain),
synced from gpu2.

- `summary_table.md` — MeanIoU pivot (LUDB / QTDB / ISP / Zhejiang) + cross-domain
- `summary_table.csv` — per-seed rows
- `summary.json` — nested aggregate
- `completion_checklist.md` — per-cell status (51/51)

Cross-domain result dirs:

- `../resnet18_mean_teacher_boundary_unet_cross_domain_merged_seed{0,1,2}/`

Regenerate after pulling more results:

```bash
python baseline/aggregate_benchmark_table.py
```
