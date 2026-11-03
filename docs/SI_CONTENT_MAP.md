# Supplementary Information map

The SI contains three Notes, four figures and eight tables. CSV paths below are relative to `data/processed/`.

| Note | Content |
|---|---|
| 1 | Rule counts, sampling, implementation checks and class boundaries |
| 2 | Single-tie derivation and matched branches |
| 3 | Empirical reconstruction, estimation and model comparisons |

| Figure | Input | Workbook sheet |
|---|---|---|
| 1 | `phase_table_certified.csv` | `SuppFig1_tolerance` |
| 2 | `theory_all_rounds.csv` | `SuppFig2_first_order` |
| 3 | `w_profile.csv` | `SuppFig3_w_profile` |
| 4 | `reward_objectives.csv` | `SuppFig4_objectives` |

| Table | Content | Input |
|---|---|---|
| 1 | Parameter grids | `../raw/cert/cert_points.json` |
| 2 | Approximation errors | `theory_endgame.csv` |
| 3 | Matched branches and ablations | `mechanism_branches.csv` |
| 4 | Sample flow | `sample_flow.csv` |
| 5 | Model-variant placements | `placement_points_classes.csv` |
| 6 | Parameter replacements | `decomposition.csv` |
| 7 | Planning objectives | `reward_objectives.csv`, `empirical_multiobjective.csv` |
| 8 | Held-out predictions | `oos_table.csv`, `oos_summary.csv` |

`analysis/make_tables.py` generates `si_tables.tex` and the eight individual table fragments. Additional saved results, quadrature checks and full trajectories remain in the data directories and the 25-sheet source-data workbook. Pilot records and their replay script are in `llm_pilot/`.
