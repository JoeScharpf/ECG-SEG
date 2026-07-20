# ECG Delineation Model — Next Steps

Not all proposed additions are worth pursuing right now. For a minimal ECG delineation model, prioritize only the additions that directly address likely failure modes.

## Add now

### 1. Constrained decoding / duplicate handling

This is more important than stochastic decoding. The output has a known grammar:

```
[start, end, class]
```

Prevent invalid sequences by restricting which token types are legal at each position, and rejecting or merging:

- `end < start`
- duplicate segments
- heavily overlapping segments of the same class
- segments generated out of chronological order

Some of this can live inside `SegmentTokenizer.batch_decode()` rather than the model itself.

### 2. Segment confidence scores

Useful for evaluation and downstream use. A simple confidence score is the mean of the log-probabilities of the tokens describing one segment:

```
score(segment) = (1/3) * [log P(s) + log P(e) + log P(c)]
```

Low-confidence segments could then be filtered, or errors could be analyzed against confidence. `generate()` currently only returns token IDs, so the selected token probabilities would need to be retained as well.

## Consider later

### 3. Sequence augmentation

May help if the model generates:

- duplicates
- false segments
- too many segments
- premature EOS tokens

Pix2Seq's original noise-object augmentation was designed around noisy bounding-box proposals; ECG segments have different structure, so it shouldn't be copied mechanically.

An ECG-specific version could add corrupted segment prefixes during training:

- shifted start or end positions
- duplicated intervals
- random intervals
- wrong wave classes

The model could learn to mark these as invalid or noise — but train the simpler model first and inspect its errors before adding this.

### 4. Pretraining and fine-tuning

Could improve results, but this is an experimental strategy rather than a missing architectural component. Possible approaches:

- pretraining the 1-D encoder on a larger ECG dataset
- masked-signal reconstruction
- heartbeat or rhythm classification
- pretraining Pix2Seq on one delineation dataset and fine-tuning on LUDB

Worth exploring after the baseline works.

## Probably do not add initially

**Stochastic decoding** — For ECG delineation, deterministic and reproducible boundaries are usually preferable. Greedy decoding is a sensible baseline. Beam search may be worth testing, but nucleus sampling could introduce unnecessary variability.

**The exact original Pix2Seq encoder** — Not needed. A 1-D ResNet is much more appropriate for ECG signals than copying Pix2Seq's image encoder.

**Traditional suppression such as NMS** — Image-style non-maximum suppression probably isn't needed. A simple temporal merging or duplicate-removal rule is more appropriate.

## Recommended order

1. Explicit positional encoding for the ECG memory
2. Constrained decoding and robust tokenizer validation
3. Confidence scores
4. Baseline training and error analysis
5. ECG-specific sequence augmentation, only if duplicates or early EOS are common
6. Pretraining, if baseline performance is limited

## Summary

The current model is sufficient for a legitimate baseline. The only additions to treat as near-essential are positional encoding, output constraints, and careful malformed-sequence handling.
