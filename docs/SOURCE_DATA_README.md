# Source Data guide

`Source_Data.xlsx` contains 25 sheets: README plus 24 data sheets. The workbook and its numerical values are preserved from the current manuscript package.

The first row of each data sheet contains the field names. `source_data_fields.csv` lists every sheet, source CSV and field in column order. Complete field construction and statistical definitions are in `analysis/build_tables.py`; workbook selections are in `analysis/make_source_data.py`, both in this repository. The manuscript Methods and figure captions define the scientific quantities and reported uncertainties. Blank cells remain blank as in the supplied data.

The source paths below are relative to the repository root. Archived sheets preserve comparisons that are available as underlying data but are not additional SI display items.

| Sheet | Rows | Source under data/processed/ |
|---|---:|---|
| `Fig1_ab_main_plane` | 208 | `phase_table_certified.csv [grid=kw; KW_COLS]` |
| `Fig1_c_slice_w0` | 16 | `slice_w0.csv` |
| `Fig1_d_tolerance` | 54 | `tolerance_transitions.csv` |
| `Fig2_a_theory` | 112 | `theory_endgame.csv` |
| `Fig2_bcd_branches` | 196 | `mechanism_branches.csv` |
| `Fig3_game_plane` | 507 | `game_plane.csv` |
| `Fig3_rules` | 10 | `game_plane_rules.csv` |
| `Fig4_robustness` | 637 | `robustness_strips.csv` |
| `Fig4_size_network` | 128 | `size_network_stats.csv` |
| `Fig4_horizon` | 112 | `horizon_window.csv` |
| `Fig5_a_estimates` | 38 | `estimates.csv` |
| `Fig5_b_kslice` | 64 | `empirical_kslice.csv` |
| `Fig5_c_decomposition` | 46 | `decomposition.csv` |
| `Fig5_d_objectives` | 10 | `empirical_multiobjective.csv` |
| `Fig5_e_prediction` | 7 | `oos_table.csv` |
| `Table1_fits` | 46 | `fits.csv` |
| `Table1_tests` | 7 | `tests.csv` |
| `SuppFig2_first_order` | 1568 | `theory_all_rounds.csv` |
| `SuppFig1_tolerance` | 208 | `phase_table_certified.csv [grid=kw; KW_COLS]` |
| `SuppFig4_objectives` | 210 | `reward_objectives.csv` |
| `Archive_quadrature` | 67 | `quadrature_nll.csv` |
| `SuppFig3_w_profile` | 33 | `w_profile.csv` |
| `Archive_branches` | 2940 | `branch_trajectories.csv` |
| `Archive_dense_boundary` | 72 | `dense_boundary.csv` |

For the two main-plane selections, the selected columns are specified by `KW_COLS` in `make_source_data.py`. All other sheets retain the source CSV column order.
