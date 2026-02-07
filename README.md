# ILI Alignment & Growth (MVP)

Two-run weld-anchored alignment + anomaly matching + growth rates. Deterministic, explainable, engineer-aligned.

## MVP: What It Does

Given two ILI runs (e.g. 2007 & 2015 OR 2015 & 2022):

1. **Ingest** – Excel/CSV
2. **Normalize** – Schema + units (distances → m, clock → degrees, depth → %, length/width → mm)
3. **Align** – Weld-anchored, segment-wise linear mapping
4. **Match** – Distance + clock + type + geometry (hard filters, weighted score)
5. **Growth** – Depth/length/width rates for matched pairs only
6. **Flag** – New, Missing, Ambiguous
7. **Output** – Excel + CSV + summary.json

Prediction is **not** part of MVP. Use `--mode stretch` to include prediction.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Run (MVP default)

```bash
python run_ili.py --mode mvp --runs 2015 2022 --input data/ILIDataV2.xlsx --output-dir output/
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--mode` | `mvp` (default), `multirun`, or `stretch` |
| `--runs` | Exactly 2 for mvp (e.g. `2007 2015` or `2015 2022`); 3 optional for multirun/stretch |
| `--input`, `-i` | Path to Excel |
| `--output-dir`, `-o` | Output directory |

- **mvp**: Two runs – normalize, align, match, growth, outputs. No prediction.
- **multirun**: Three runs – chain alignment + tracks. No prediction.
- **stretch**: Includes prediction + risk flags.

## Outputs

- `output/output.xlsx` – All tables as sheets
- `output/tables/*.csv` – One CSV per table
- `output/tables/summary.json` – Counts, match rate %, growth stats, flags
- `output/tables/schema_report.json` – Column mapping, units

## Configuration

Edit `config.py` for tolerances, weights, and growth sanity:

- `AXIAL_TOL_M`, `CLOCK_TOL_DEG`, `MATCH_SCORE_EPSILON`
- `W_DIST`, `W_CLOCK`, `W_DEPTH`, `W_LENGTH`, `W_WIDTH`
- `MAX_DEPTH_RATE_OUTLIER`, `SEGMENT_LENGTH_MIN_M`
