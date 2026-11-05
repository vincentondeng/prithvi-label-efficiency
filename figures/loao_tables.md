# Leave-Africa-out analysis

LOAO training pool: non-African chips only; chips per run by fraction: 10% = 20, 50% = 98, 100% = 196. African evaluation chips per run: 97.

## A. Transfer to Africa: pretrained vs scratch, models trained without any African chip

| labels (chips) | evaluation set | pretrained | scratch | gap pre - scr, 95% t-CI (wins) |
|---|---|---|---|---|
| 10% (20) | Africa, all 97 chips | 0.749 ± 0.034 | 0.726 ± 0.022 | +0.023 [+0.001, +0.044] (4/5) |
| 10% (20) |   Ghana | 0.663 ± 0.038 | 0.656 ± 0.019 | +0.007 [-0.032, +0.047] (4/5) |
| 10% (20) |   Nigeria | 0.861 ± 0.047 | 0.843 ± 0.036 | +0.018 [-0.013, +0.049] (4/5) |
| 10% (20) |   Somalia | 0.632 ± 0.055 | 0.579 ± 0.047 | +0.053 [-0.001, +0.106] (5/5) |
| 10% (20) | Africa, test chips only | 0.686 ± 0.031 | 0.659 ± 0.029 | +0.027 [-0.000, +0.054] (4/5) |
| 10% (20) | non-African test chips | 0.818 ± 0.026 | 0.778 ± 0.035 | +0.040 [+0.024, +0.056] (5/5) |
| 50% (98) | Africa, all 97 chips | 0.803 ± 0.005 | 0.786 ± 0.006 | +0.016 [+0.008, +0.024] (5/5) |
| 50% (98) |   Ghana | 0.695 ± 0.004 | 0.689 ± 0.017 | +0.007 [-0.017, +0.031] (4/5) |
| 50% (98) |   Nigeria | 0.899 ± 0.003 | 0.886 ± 0.005 | +0.013 [+0.010, +0.017] (5/5) |
| 50% (98) |   Somalia | 0.734 ± 0.008 | 0.695 ± 0.022 | +0.039 [+0.013, +0.065] (5/5) |
| 50% (98) | Africa, test chips only | 0.722 ± 0.010 | 0.710 ± 0.017 | +0.012 [-0.002, +0.025] (4/5) |
| 50% (98) | non-African test chips | 0.857 ± 0.006 | 0.823 ± 0.008 | +0.034 [+0.024, +0.044] (5/5) |
| 100% (196) | Africa, all 97 chips | 0.811 ± 0.002 | 0.780 ± 0.005 | +0.032 [+0.025, +0.038] (5/5) |
| 100% (196) |   Ghana | 0.707 ± 0.008 | 0.677 ± 0.006 | +0.030 [+0.017, +0.042] (5/5) |
| 100% (196) |   Nigeria | 0.906 ± 0.001 | 0.889 ± 0.002 | +0.017 [+0.013, +0.021] (5/5) |
| 100% (196) |   Somalia | 0.741 ± 0.006 | 0.667 ± 0.021 | +0.074 [+0.042, +0.106] (5/5) |
| 100% (196) | Africa, test chips only | 0.730 ± 0.001 | 0.696 ± 0.004 | +0.033 [+0.027, +0.040] (5/5) |
| 100% (196) | non-African test chips | 0.859 ± 0.003 | 0.830 ± 0.004 | +0.028 [+0.021, +0.036] (5/5) |

## B. Value of in-region labels: African TEST chips (21), trained with vs without African chips

Penalty = IoU(with Africa) - IoU(without Africa), paired by seed. The training sets also differ in size (with: 25 / 252 chips; without: 20 / 196), so the penalty mixes 'no African labels' with 'fewer labels'.

| labels | arm | with Africa | without Africa | penalty, 95% t-CI (seeds with penalty > 0) |
|---|---|---|---|---|
| 10% | pretrained | 0.699 ± 0.015 | 0.686 ± 0.031 | +0.014 [-0.025, +0.052] (4/5) |
| 10% | scratch | 0.671 ± 0.021 | 0.659 ± 0.029 | +0.012 [-0.015, +0.040] (4/5) |
| 100% | pretrained | 0.725 ± 0.005 | 0.730 ± 0.001 | -0.005 [-0.011, +0.001] (1/5) |
| 100% | scratch | 0.690 ± 0.005 | 0.696 ± 0.004 | -0.007 [-0.015, +0.001] (1/5) |

Does pre-training reduce the penalty? (penalty_scratch - penalty_pretrained; > 0 means yes)

| labels | difference, 95% t-CI (seeds where pre-training reduces the penalty) |
|---|---|
| 10% | -0.001 [-0.039, +0.037] (2/5) |
| 100% | -0.002 [-0.012, +0.008] (2/5) |

## C. Sensitivity S1: without chip Ghana_866994 (post hoc; see quicklook figure)

| labels | evaluation set | pretrained | scratch | gap pre - scr (wins) | gap with all chips |
|---|---|---|---|---|---|
| 10% | Africa, all chips | 0.764 ± 0.034 | 0.741 ± 0.023 | +0.024 [+0.004, +0.043] (5/5) | +0.023 |
| 10% |   Ghana | 0.703 ± 0.036 | 0.696 ± 0.016 | +0.007 [-0.030, +0.044] (3/5) | +0.007 |
| 10% | Africa, test chips | 0.763 ± 0.028 | 0.726 ± 0.033 | +0.038 [+0.005, +0.070] (5/5) | +0.027 |
| 50% | Africa, all chips | 0.822 ± 0.005 | 0.804 ± 0.006 | +0.018 [+0.010, +0.026] (5/5) | +0.016 |
| 50% |   Ghana | 0.743 ± 0.007 | 0.733 ± 0.014 | +0.010 [-0.012, +0.033] (4/5) | +0.007 |
| 50% | Africa, test chips | 0.816 ± 0.007 | 0.794 ± 0.017 | +0.022 [+0.006, +0.038] (5/5) | +0.012 |
| 100% | Africa, all chips | 0.831 ± 0.002 | 0.799 ± 0.005 | +0.032 [+0.026, +0.038] (5/5) | +0.032 |
| 100% |   Ghana | 0.755 ± 0.009 | 0.726 ± 0.004 | +0.029 [+0.018, +0.040] (5/5) | +0.030 |
| 100% | Africa, test chips | 0.824 ± 0.003 | 0.788 ± 0.003 | +0.036 [+0.031, +0.041] (5/5) | +0.033 |

## D. Weight of each event in the pooled 'Africa, all 97 chips' IoU (union share; mean over runs)

| labels | arm | Ghana | Nigeria | Somalia |
|---|---|---|---|---|
| 10% | pretrained | 0.31 | 0.47 | 0.22 |
| 10% | scratch | 0.30 | 0.47 | 0.23 |
| 50% | pretrained | 0.31 | 0.49 | 0.20 |
| 50% | scratch | 0.31 | 0.49 | 0.20 |
| 100% | pretrained | 0.31 | 0.49 | 0.20 |
| 100% | scratch | 0.31 | 0.49 | 0.20 |

Ghana_866994 share of Ghana's labelled water over the 97 chips: 0.18
