# Copyright (c) ECG-SEG. Read-only diagnostics for the Pix2Seq ECG model (plan Step 1-2).
#
# Loads a trained checkpoint and reports the metrics that decide the next action:
#   - ground-truth segment-count distribution + truncation fraction at max_segments
#   - predicted vs GT segment counts (free-running decode)
#   - valid-triple rate
#   - teacher-forced vs free-running token accuracy by token type (exposure error)
#   - per-class IoU (esp. P-wave), with constrained vs unconstrained decoding
#   - coordinate accuracy within +/- 1/3/5/10 bins (segments matched by max IoU)
#   - EOS position distribution
#
# This script only reads data and prints; it does not train or modify anything.

import argparse
import os
from collections import defaultdict

import mergedeep
import numpy as np
import torch
import yaml

from models.pix2seq.model import build_pix2seq_from_cfg
from utils.semi_dataset import build_seg_dataset, get_dataloader


def parse() -> dict:
    parser = argparse.ArgumentParser("Pix2Seq ECG diagnostics (read-only)")
    parser.add_argument("-f", "--config_path", required=True, type=str)
    parser.add_argument("-o", "--override_config_path", default=None, type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--exp_name", default="", type=str)
    parser.add_argument("--model_path", default="", type=str)
    parser.add_argument("--split", default="valid", type=str, help="valid|test")
    parser.add_argument("--max_batches", default=0, type=int, help="0 = all batches")
    parser.add_argument("--device", default=None, type=str)
    args = parser.parse_args()

    with open(os.path.realpath(args.config_path), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if args.override_config_path:
        with open(os.path.realpath(args.override_config_path), "r") as f:
            config = mergedeep.merge(config, yaml.load(f, Loader=yaml.FullLoader))
    for k in ("output_dir", "exp_name"):
        v = getattr(args, k)
        if v:
            config[k] = v
    config["_args"] = vars(args)
    return config


def interval_iou(a, b):
    """IoU of two inclusive intervals (start, end)."""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union > 0 else 0.0


def match_segments(pred, gt):
    """Greedy same-class max-IoU matching. Returns list of (pred_seg, gt_seg)."""
    matches = []
    used = set()
    by_class = defaultdict(list)
    for j, g in enumerate(gt):
        by_class[g[0]].append((j, g))
    for p in pred:
        best, best_iou = None, 0.0
        for j, g in by_class[p[0]]:
            if j in used:
                continue
            iou = interval_iou((p[1], p[2]), (g[1], g[2]))
            if iou > best_iou:
                best, best_iou = (j, g), iou
        if best is not None and best_iou > 0:
            used.add(best[0])
            matches.append((p, best[1]))
    return matches


def percentiles(values):
    if not values:
        return {}
    a = np.asarray(values)
    return {
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": int(a.max()),
        "mean": float(a.mean()),
    }


def report_truncation(config, tokenizer):
    print("\n=== Truncation (GT segment-count distribution) ===")
    for split in ("train_labeled", "valid"):
        try:
            ds = build_seg_dataset(config["dataset"], split=split)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{split}] skipped ({exc})")
            continue
        counts = []
        for i in range(len(ds)):
            label = ds[i]["target"]
            counts.append(len(tokenizer.mask_to_segments(label)))
        frac_trunc = float(np.mean([c > tokenizer.max_segments for c in counts]))
        pct = percentiles(counts)
        print(
            f"  [{split}] n={len(counts)}  "
            f"segments {pct}  frac>max_segments({tokenizer.max_segments})={frac_trunc:.3f}"
        )


def token_type(tok, model):
    if tok == model.pad_id:
        return "pad"
    if tok == model.bos_id:
        return "bos"
    if tok == model.eos_id:
        return "eos"
    if model.class_lo <= tok < model.class_hi:
        return "class"
    if model.coord_lo <= tok < model.coord_hi:
        return "coord"
    return "other"


def valid_triple_rate(tokens, model):
    """Fraction of triple-slots that are well-formed (class, coord, coord)."""
    cleaned = []
    for t in tokens:
        t = int(t)
        if t in (model.pad_id, model.bos_id):
            continue
        if t == model.eos_id:
            break
        cleaned.append(t)
    if not cleaned:
        return 1.0, 0  # nothing emitted -> vacuously valid, 0 triples
    groups = len(cleaned) // 3
    if groups == 0:
        return 0.0, 0
    valid = 0
    for g in range(groups):
        c, s, e = cleaned[3 * g], cleaned[3 * g + 1], cleaned[3 * g + 2]
        if (
            model.class_lo <= c < model.class_hi
            and model.coord_lo <= s < model.coord_hi
            and model.coord_lo <= e < model.coord_hi
        ):
            valid += 1
    return valid / groups, groups


def main():
    config = parse()
    args = config["_args"]
    device = torch.device(
        args["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = build_pix2seq_from_cfg(config)
    tokenizer = model.tokenizer

    # Truncation stats do not need the checkpoint.
    report_truncation(config, tokenizer)

    output_dir = os.path.join(config.get("output_dir", ""), config.get("exp_name", ""))
    ckpt = args["model_path"] or os.path.join(
        output_dir, f"best-{config.get('test', {}).get('target_metric', 'MeanIoU')}.pth"
    )
    assert os.path.exists(ckpt), f"Checkpoint not found: {ckpt}"
    state = torch.load(ckpt, map_location="cpu")["model"]
    print(f"\nLoaded checkpoint: {ckpt}")
    print(model.load_state_dict(state, strict=False))
    model.to(device).eval()

    ds = build_seg_dataset(config["dataset"], split=args["split"])
    loader = get_dataloader(
        ds, is_distributed=False, mode="eval", **config["dataloader"]
    )

    bin_width = tokenizer.signal_length / tokenizer.num_bins
    tol_bins = [1, 3, 5, 10]

    for constrained in (False, True):
        tag = "constrained" if constrained else "unconstrained"
        print(f"\n=== Free-running decode diagnostics ({tag}) ===")
        n_samples = 0
        pred_count_sum = 0
        gt_count_sum = 0
        vtr_sum = 0.0
        eos_positions = []
        onset_err = []
        offset_err = []
        inter = np.zeros(model.num_classes)
        union = np.zeros(model.num_classes)

        # Teacher-forced accuracy accumulators (decode-independent; compute once).
        tf_correct = defaultdict(int)
        tf_total = defaultdict(int)

        for bi, samples in enumerate(loader):
            if args["max_batches"] and bi >= args["max_batches"]:
                break
            inputs = samples["ecg"].to(device, non_blocking=True)
            labels = samples["target"]
            if labels.dim() == 3:
                labels = labels.squeeze(1)
            labels = labels.to(device)

            with torch.no_grad():
                memory = model.encode(inputs)
                gen = model.generate(memory, constrained=constrained)

                if not constrained:
                    # Teacher-forced token accuracy by type (only needs one pass).
                    tgt = tokenizer.batch_encode(labels, device=device)
                    logits = model.forward_tokens(memory, tgt[:, :-1])
                    pred_tok = logits.argmax(dim=-1)
                    tgt_out = tgt[:, 1:]
                    for r in range(tgt_out.size(0)):
                        for c in range(tgt_out.size(1)):
                            gold = int(tgt_out[r, c].item())
                            if gold == model.pad_id:
                                continue
                            tt = token_type(gold, model)
                            tf_total[tt] += 1
                            tf_correct[tt] += int(pred_tok[r, c].item() == gold)

            gen = gen.cpu()
            labels_cpu = labels.cpu()
            for i in range(gen.size(0)):
                toks = gen[i].tolist()
                pred_segs = tokenizer.tokens_to_segments(toks)
                gt_segs = tokenizer.mask_to_segments(labels_cpu[i])
                n_samples += 1
                pred_count_sum += len(pred_segs)
                gt_count_sum += len(gt_segs)
                vtr, _ = valid_triple_rate(toks, model)
                vtr_sum += vtr
                if model.eos_id in toks:
                    eos_positions.append(toks.index(model.eos_id))

                # Per-class IoU on rasterized masks.
                pred_mask = tokenizer.segments_to_mask(pred_segs)
                gt_mask = labels_cpu[i].numpy().astype(int)
                for c in range(model.num_classes):
                    p = pred_mask == c
                    g = gt_mask == c
                    inter[c] += np.logical_and(p, g).sum()
                    union[c] += np.logical_or(p, g).sum()

                # Boundary error on matched segments.
                for p_seg, g_seg in match_segments(pred_segs, gt_segs):
                    onset_err.append(abs(p_seg[1] - g_seg[1]))
                    offset_err.append(abs(p_seg[2] - g_seg[2]))

        ious = [inter[c] / union[c] if union[c] > 0 else float("nan") for c in range(model.num_classes)]
        mean_iou = float(np.nanmean(ious))
        print(f"  samples={n_samples}")
        print(f"  pred segments/sample={pred_count_sum / max(n_samples,1):.2f}  "
              f"gt segments/sample={gt_count_sum / max(n_samples,1):.2f}")
        print(f"  valid-triple rate={vtr_sum / max(n_samples,1):.3f}")
        print(f"  per-class IoU={[round(x,3) for x in ious]}  MeanIoU={mean_iou:.3f}")
        if eos_positions:
            print(f"  EOS position: {percentiles(eos_positions)}")
        else:
            print("  EOS position: never emitted")
        if onset_err:
            oe = np.asarray(onset_err)
            fe = np.asarray(offset_err)
            print(f"  matched segments={len(onset_err)}  "
                  f"onset MAE={oe.mean():.1f} samples ({oe.mean()/bin_width:.2f} bins)  "
                  f"offset MAE={fe.mean():.1f} samples ({fe.mean()/bin_width:.2f} bins)")
            for k in tol_bins:
                thr = k * bin_width
                print(f"    within +/-{k} bins: onset={np.mean(oe<=thr):.3f}  offset={np.mean(fe<=thr):.3f}")
        else:
            print("  matched segments=0 (no overlap between predicted and GT segments)")

        if not constrained:
            print("  teacher-forced token accuracy by type:")
            for tt in ("class", "coord", "eos"):
                if tf_total[tt]:
                    print(f"    {tt}: {tf_correct[tt]/tf_total[tt]:.3f}  (n={tf_total[tt]})")


if __name__ == "__main__":
    main()
