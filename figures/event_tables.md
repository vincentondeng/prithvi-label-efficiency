## Africa vs rest, test water IoU (pixel-pooled; Africa = Ghana, Nigeria, Somalia)

Chips in this split: 21 African, 69 other.

| labels | group | pretrained | scratch | gap (pre - scr), 95% t-CI (wins) |
|---|---|---|---|---|
| 10% (n=5) | africa | 0.699 ± 0.015 | 0.671 ± 0.021 | +0.028 [+0.005, +0.051] (5/5) |
| 10% (n=5) | rest | 0.815 ± 0.012 | 0.782 ± 0.010 | +0.033 [+0.016, +0.049] (5/5) |
| 10% (n=5) | all | 0.797 ± 0.011 | 0.765 ± 0.011 | +0.032 [+0.016, +0.048] (5/5) |
| 100% (n=5) | africa | 0.725 ± 0.005 | 0.690 ± 0.005 | +0.035 [+0.027, +0.043] (5/5) |
| 100% (n=5) | rest | 0.865 ± 0.003 | 0.828 ± 0.004 | +0.037 [+0.031, +0.042] (5/5) |
| 100% (n=5) | all | 0.843 ± 0.002 | 0.807 ± 0.004 | +0.036 [+0.032, +0.041] (5/5) |

### Is the pretrained gap different in Africa than elsewhere? (gap_Africa - gap_rest, paired by seed)

| labels | gap_Africa - gap_rest, 95% t-CI (seeds where Africa gap larger) |
|---|---|
| 10% | -0.005 [-0.025, +0.015] (3/5) |
| 100% | -0.001 [-0.012, +0.010] (2/5) |

### Per-event water IoU (mean over seeds; descriptive, few chips per event)

| event (chips) | pre 10% | scr 10% | pre 100% | scr 100% |
|---|---|---|---|---|
| Ghana * (11) | 0.509 | 0.501 | 0.533 | 0.499 |
| India (14) | 0.872 | 0.847 | 0.899 | 0.865 |
| Mekong (6) | 0.914 | 0.892 | 0.931 | 0.902 |
| Nigeria * (4) | 0.927 | 0.909 | 0.939 | 0.922 |
| Pakistan (6) | 0.592 | 0.616 | 0.755 | 0.700 |
| Paraguay (14) | 0.695 | 0.634 | 0.732 | 0.692 |
| Somalia * (6) | 0.766 | 0.691 | 0.812 | 0.750 |
| Spain (6) | 0.812 | 0.764 | 0.849 | 0.805 |
| Sri-Lanka (9) | 0.854 | 0.811 | 0.932 | 0.914 |
| USA (14) | 0.788 | 0.749 | 0.816 | 0.750 |

`*` = African event.

### Weight of each African event in the pooled Africa IoU (w_e = union_e / sum of unions; mean over seeds)

| labels | arm | Ghana | Nigeria | Somalia |
|---|---|---|---|---|
| 10% | pretrained | 0.45 | 0.31 | 0.24 |
| 10% | scratch | 0.45 | 0.30 | 0.25 |
| 100% | pretrained | 0.45 | 0.31 | 0.24 |
| 100% | scratch | 0.45 | 0.31 | 0.24 |

### African training chips per seed (same subset for both arms) and per-event test IoU

| labels | seed | train chips: Ghana / Nigeria / Somalia | IoU pre: Ghana / Nigeria / Somalia | IoU scr: Ghana / Nigeria / Somalia |
|---|---|---|---|---|
| 10% | 0 | 3 / 3 / 3 | 0.496 / 0.936 / 0.802 | 0.497 / 0.921 / 0.744 |
| 10% | 1 | 6 / 2 / 0 | 0.498 / 0.921 / 0.748 | 0.444 / 0.912 / 0.653 |
| 10% | 2 | 2 / 1 / 1 | 0.526 / 0.932 / 0.795 | 0.517 / 0.917 / 0.721 |
| 10% | 3 | 3 / 2 / 1 | 0.466 / 0.920 / 0.780 | 0.511 / 0.893 / 0.715 |
| 10% | 4 | 2 / 1 / 0 | 0.560 / 0.926 / 0.707 | 0.536 / 0.902 / 0.622 |
| 100% | 0 | 31 / 10 / 15 | 0.548 / 0.941 / 0.813 | 0.489 / 0.923 / 0.765 |
| 100% | 1 | 31 / 10 / 15 | 0.531 / 0.939 / 0.811 | 0.507 / 0.924 / 0.765 |
| 100% | 2 | 31 / 10 / 15 | 0.529 / 0.939 / 0.811 | 0.509 / 0.920 / 0.731 |
| 100% | 3 | 31 / 10 / 15 | 0.518 / 0.938 / 0.815 | 0.491 / 0.921 / 0.745 |
| 100% | 4 | 31 / 10 / 15 | 0.537 / 0.938 / 0.812 | 0.497 / 0.920 / 0.746 |
