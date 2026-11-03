# Network planner phase diagrams

Code and data for *Behavioural calibration and planning objectives determine when network planners should conciliate defectors*.

## Analysis

Python 3.11.16:

```sh
python -m pip install -r requirements-analysis.txt
python analysis/run_analysis.py
python -m unittest discover -s tests -v
python llm_pilot/replay_records.py
```

The analysis rebuilds processed data, figures, LaTeX tables and `Source_Data.xlsx` from the saved outputs in `data/raw/`. See the [output map](docs/REPRODUCTION_MAP.md), [SI map](docs/SI_CONTENT_MAP.md) and [Source Data guide](docs/SOURCE_DATA_README.md).

## Simulation

Python 3.11.10 on Linux/WSL:

```sh
python -m pip install -r requirements-simulation.txt
python code/ph_cert.py out_cert 4
```

The last argument sets the worker count. Other entry points document their arguments in `code/`. Empirical reconstruction uses `8ahkg-osfstorage-archive.zip` from [OSF](https://osf.io/8ahkg/) (SHA-256: `6a6afb8cf03d8ba5e7145a53ac6834956a58bf042be5408ba5c2015233a5b3bf`).

Code: [MIT](LICENSE_code.txt). Data: CC BY 4.0; source provenance is recorded in `data/legacy/` and the Source Data guide.
