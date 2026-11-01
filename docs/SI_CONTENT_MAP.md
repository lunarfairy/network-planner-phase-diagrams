# Supplementary Information content map

The compact SI contains four Notes, four figures and nine tables. The saved raw and processed values are unchanged. The presentation generators select the compact displays; the underlying data retain the complete comparisons.

## Notes

| Current | Previous | Content and complete data |
|---|---|---|
| Note 1 | Notes 1-2 | Horizon-dependent rule count, seeds, implementation comparison and dense boundaries; `replicates_default.csv`, `tolerance_transitions.csv`, `dense_boundary.csv`, `data/raw/legacy/simulator_validation.csv` |
| Note 2 | Notes 3-4 | Linearisation, isolated-node convention, paired branch contributions and ablations; `theory_endgame.csv`, `theory_all_rounds.csv`, `mechanism_branches.csv`, `branch_trajectories.csv` |
| Note 3 | Notes 5, 6 and 8 | Coefficient assignments, reconstruction, current quadrature and profile, condition parameters and additional predictions; `sample_flow.csv`, `quadrature_nll.csv`, `quadrature_fits.csv`, `w_profile.csv`, `w_profile_summary.csv`, `estimates.csv`, `fits.csv`, `tests.csv`, `placement_points_classes.csv`, `placement_bootstrap.csv`, `decomposition.csv`, `oos_table.csv` |
| Note 4 | Note 10 | Language-model pilot methods; `llm_pilot/` retains the saved records and prompt-generating code |

Previous Note 7's objective definitions are in current Table 8's caption. Previous Note 9's threshold-search protocol is in the main Fig. 3 caption; all candidates remain in `game_plane_rules.csv` and the `Fig3_rules` workbook sheet. CSV paths without a prefix refer to `data/processed/`, generated from the saved raw data.

## Figures

| Current | Previous | Change | Processed input / workbook sheet |
|---|---|---|---|
| Fig. 1 | Fig. 2 | Three panels: two alternative tolerances and raw argmax; default tolerance is in main Fig. 1 | `phase_table_certified.csv` / `SuppFig1_tolerance` |
| Fig. 2 | Fig. 1 | Renumbered; display unchanged | `theory_all_rounds.csv` / `SuppFig2_first_order` |
| Fig. 3 | Fig. 4b | All three temperature profiles; quadrature now in Table 5 | `w_profile.csv` / `SuppFig3_w_profile` |
| Fig. 4 | Fig. 3 | Renumbered; display unchanged | `reward_objectives.csv` / `SuppFig4_objectives` |
| Data archive | Fig. 5 | Full trajectories remain available | `branch_trajectories.csv` / `Archive_branches` |

The analysis also exports `Archive_branch_trajectories` for the full trajectories. The supplementary ZIP preserves the previous figure PDFs and PNGs under `figures/archive_pre_compaction/`, and the previous SI text and tables under `manuscript/archive_pre_compaction/`.

## Tables

| Current | Previous | Content / selection |
|---|---|---|
| Table 1 | Table 1 | All parameter grids |
| Table 2 | Table 5 | Approximation errors |
| Table 3 | Table 10 | All matched-branch and ablation rows |
| Table 4 | Table 6 | Condition, groups, players, bot/human decisions, capital exclusions and analysed decisions |
| Table 5 | Table 7 | Current M1/M3 at 96/160/320 nodes and M1 refits at 160/320 nodes |
| Table 6 | Table 11 | All 16 placement rows; combined A/B classes; evaluation gain on B and conciliation gain on A explicitly labelled |
| Table 7 | Table 12 | All 23 parameter-replacement rows, including both directions and discordant A/B classes |
| Table 8 | Table 13 | All four alternative objectives; discount and rejection definitions in the caption |
| Table 9 | Table 14 | All seven conditions, observed values, published model, LOCO M1/M3 with M3 intervals, and both MAE summaries |

Previous Tables 2, 3, 4, 8, 9, 15 and 16 are represented by the complete source tables listed above. Earlier quadrature comparisons remain in `quadrature_nll.csv` and `quadrature_fits.csv`; the public repository additionally records their provenance in `data/legacy/`. The workbook retains 25 sheets (README plus 24 data sheets), including `Archive_quadrature`, `Archive_branches` and `Archive_dense_boundary`.

This revision changes presentation and cross-references. It does not rerun simulation or estimation.

The manuscript places figures and tables beside their relevant discussion. `analysis/make_tables.py` emits both `si_tables.tex` and the nine `si_table1.tex` through `si_table9.tex` fragments used at the inline insertion points.
