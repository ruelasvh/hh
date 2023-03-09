# Plot HH4b Analysis

## Installation
```bash
git clone ssh://git@gitlab.cern.ch:7999/viruelas/plot-hh4b-analysis.git
cd plot-hh4b-analysis
pip install -e .
```

## Usage
```bash
hh4b_non_res_res_make_hists inputs.json
```

For example, `inputs.json`
```json
{
    "k01": "path/to/k01inputsdir/",
    "k10": "path/to/k10inputsdir/",
    "dijets": "..."
}
```