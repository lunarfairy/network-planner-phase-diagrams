# Network planner phase diagrams

Code and data for *Behavioural calibration and planning objectives determine when network planners should conciliate defectors*.

## Analysis

Use Python 3.11.16 and run from the repository root:

```sh
python -m pip install -r requirements-analysis.txt
python analysis/run_analysis.py
```

The analysis reads `data/raw/` and generates processed tables, figures, LaTeX table fragments and `Source_Data.xlsx`. See the [figure and input map](docs/REPRODUCTION_MAP.md) for individual outputs.

Saved pilot records and data checks:

```sh
python llm_pilot/replay_records.py
python -m unittest discover -s tests -v
```

With a writable manuscript source directory, `python analysis/run_analysis.py --ms /path/to/manuscript` also writes its table fragments and checks decimal tokens against the stored number index. The directory must contain `manuscript.tex` and `supplementary_information.tex`.

## Simulation and estimation

Create a separate Python 3.11.10 environment:

```sh
python -m pip install -r requirements-simulation.txt
python code/ph_cert.py out_cert 4
```

Full simulation and estimation entry points use Unix `fork` multiprocessing; run them on Linux or WSL. The final argument above selects four workers. The sweep evaluates 50 planner rules and two reference planners across 1,352 parameter points, with two independent samples of 10,000 games per point. Per-script usage strings specify the other run commands.

| Directory | Contents |
|---|---|
| `code/` | Simulator, parameter sweeps and empirical estimation |
| `analysis/` | Tables, figures, source data and manuscript-number checks |
| `data/raw/` | Saved simulation and estimation outputs |
| `data/legacy/` | Historical comparison rows and source provenance |
| `llm_pilot/` | Saved pilot responses and offline summary |
| `tests/` | Archived-data consistency checks |

Empirical reconstruction uses the McKee et al. data at [OSF 8ahkg](https://osf.io/8ahkg/). The input archive is `8ahkg-osfstorage-archive.zip` (2,085,362 bytes; SHA-256 `6a6afb8cf03d8ba5e7145a53ac6834956a58bf042be5408ba5c2015233a5b3bf`). The empirical entry point estimates the revised-panel models and reads the earlier-panel comparison from `data/legacy/`, where row selections and source hashes are documented.

## License

Code: [MIT](LICENSE_code.txt). Derived data: CC BY 4.0, as specified in the source archive. The version-pinned requirements files describe the analysis and simulation environments; `environment.yml` records the original general environment specification.
