# Notes on the paper

Where the logs in this repository and the camera-ready paper don't line up exactly,
and why.

## Numbers that differ

**Fig. 4b, Romania.** The script gives 79.7% vs 77.7% (+2.0 pp). The paper prints
79.8% vs 78.0% (+1.8 pp). The paper's run (July 2026) used the Romanian archive before we
trimmed the study period to 2024-07..2026-05. Its log (not shipped) showed 2,204,206
passes on the graph, against 2,179,358 now: the extra 1.1% was one isolated month,
January 2023. That month is not in the shipped data, so the paper's exact values can't be
regenerated. Germany
matches exactly. The reading is the same: the gap is small and has opposite signs in the
two countries (Germany -2.2 pp).

**Table I, Romania 4.1 M passes.** That is the row count of the whole per-pass archive,
which covers 34,654 reaches of the wider Romanian EU-Hydro network. The study graph keeps 18,692 of them,
and 2.18 M passes fall on it. So the "18.7 M observations" in Limitations is closer to
16.8 M on the study graphs. Germany's 14.6 M is exact.

## Wording worth qualifying

- **Fig. 4 caption** says "facility-adjacent exceedances", but both panels use all
  exceedances. The logs print both versions. In panel (a) the shares differ by at most
  2.4 points (Romania "concurrent": 78% all, 81% facility-adjacent). In panel (b) the
  facility-adjacent gaps are -2.5 pp (Germany) and +1.6 pp (Romania), against -2.2 and
  +2.0 for all exceedances: same signs, same reading.
- **"2,000 resamples throughout"** is true for the headline tests. Runs that repeat a
  test many times used fewer: the basin curves (200 for Germany, 300 for Romania), the
  resolution sweep (1,000 and 300), the Fig. 4 control and the inherited-share split
  (1,000; the paper quotes no interval from these two). The basin curves also
  sample fewer source reaches, so their wiggles are mostly noise. `BASINS_headline.txt`
  and `ro_basins_headline.txt` repeat them at full precision: Germany +0.018 to +0.029,
  Romania +0.027 to +0.066, every point significant.
- **The time-lag test** (Sec. V-C) is described as having "a sign unstable across
  resamples". More precisely, the sign changes across signals, countries, controls and
  bin widths. At monthly cadence it points downstream for NDCI in both countries and for
  Romanian turbidity (common-mode +0.041 [+0.004, +0.077]), with CIs above zero. It does
  not for German turbidity, for German NDCI under the bearing-matched control
  (+0.025 [-0.004, +0.053]), or for NDCI at the 15-day bin. See the H2 lines in
  `COFLUCTUATION*.txt` and `ro_cofluctuation*.txt`.
- **Gap-filling skill -0.016** (Sec. V-B) is the exceedance-flag row of `GAPFILL.txt`,
  which also lists dissolved oxygen and nitrate.
- **Table III static cells** subtract rounded correlations (0.443 - 0.541 = -0.098). The
  unrounded differences in the log are -0.099 and +0.018.
- **"Nearly half of all upstream reaches already sit near some facility"** (Sec. V-B)
  holds for Germany (43-47% at 1.5 km). In Romania it is 14-15%
  (`{de,ro}_attribution_upstream.txt`), which is why Romania shows the marginal edge.
- **"Delta corr 0.03-0.15"** in the Discussion is the common-mode range. With each
  reach's own seasonal cycle removed, German turbidity is +0.019 (Fig. 3).
- **Licence.** The paper says the code and derived panels are released under
  Apache-2.0. The code is. The data keeps its source terms (see `data/README.md`), and
  the OpenStreetMap facility layers are ODbL.
