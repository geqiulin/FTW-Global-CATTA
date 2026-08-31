# FTW Global CATTA

Country-adaptive spatial test-time augmentation and watershed post-processing for the [Fields of the World](https://github.com/fieldsoftheworld/ftw-baselines) benchmark.

## Result

The release evaluates the official PRUE EfficientNet-B7 checkpoint on all 20 full-data FTW countries. Country-specific post-processing configurations were selected on validation data, frozen, and then evaluated once on each country's test split.

| Scope | Pixel IoU | Object precision | Object recall | Object F1 |
|---|---:|---:|---:|---:|
| Official PRUE B7 inference | 0.830781 | 0.621423 | 0.396649 | 0.484223 |
| Global CATTA Watershed | **0.841942** | **0.710888** | **0.425276** | **0.532183** |
| Absolute Object F1 gain |  |  |  | **+0.047960** |

The `All countries` row is pooled from pixel-level and object-level TP/FP/FN counts over Austria, Belgium, Cambodia, Corsica, Croatia, Denmark, Estonia, Finland, France, Germany, Latvia, Lithuania, Luxembourg, Netherlands, Slovakia, Slovenia, South Africa, Spain, Sweden, and Vietnam.

## Method

1. Run the released PRUE EfficientNet-B7 model with four spatial views.
2. Select a country-specific probability decision rule on validation data only.
3. Split large merged components with boundary-aware watershed operations.
4. Grow conservative instance seeds using field, background, and boundary probabilities.
5. Freeze every country configuration before evaluating its test split.

No extra training data or model fine-tuning is used. The neural network has **67.098499 million parameters**; the added post-processing has no trainable parameters.

## Files

- `results/ftw_full_20_country_metrics.csv`: leaderboard submission file.
- `results/full_submission_report.json`: pooled counts and aggregate metrics.
- `reports/validation/`: validation-only configuration-selection reports.
- `reports/test/`: one frozen test report per country.
- `build_ftw_country_cache_fast.py`: PRUE B7 spatial-TTA inference cache builder.
- `optimize_ftw_country_validation.py`: validation-only country configuration search.
- `evaluate_ftw_country_frozen.py`: frozen test evaluator.
- `build_ftw_full_submission.py`: strict 20-country aggregator.
- `verify_full_release.py`: independent release consistency check.

## Verify

Install the official FTW environment and the packages in `requirements.txt`, then run:

```bash
python verify_full_release.py
```

The verifier refuses partial country sets and independently recomputes the pooled metrics from all country reports.

## Leaderboard submission

- Metrics CSV: `results/ftw_full_20_country_metrics.csv`
- Model type: `PRUE EfficientNet-B7 + Country-Adaptive Spatial-TTA Watershed`
- Parameters (millions): `67.098499`

The PRUE checkpoint is not redistributed here. Obtain it from the official FTW baseline release and verify SHA-256:

```text
2b1b34a17b85b8f70da6ff737529743b6bc6049e987bce5a1fcdd7279eb3b120
```

## License and attribution

Code is released under the MIT license. The base model, dataset tooling, and benchmark belong to the Fields of the World authors; cite the official project when using this work.
