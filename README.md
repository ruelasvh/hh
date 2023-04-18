# Plot HH4b Analysis

## Installation
```bash
git clone https://github.com/ruelasvh/plot-hh4b-analysis.git
cd plot-hh4b-analysis
pip install -e .
```

## Usage
```bash
hh4b_non_res_res_make_hists config.json
```

For example, `config.json`
```json
{
    "inputs": {
        "k01": "path/to/k01inputsdir/",
        "k10": "path/to/k10inputsdir/",
        "dijets": "..."
    },
    "event_selection": {
        "central_jets": {
            "min_pt": 40000,
            "max_eta": 2.5,
            "min_nconstituents": 4
        },
        "btagging": {
            "model": "DL1dv00",
            "efficiency": 0.7
        },
        "forward_jets": {
            "min_pt": 30000,
            "min_eta": 2.5,
            "min_nconstituents": 6
        },
        "top_veto": {
            "ggF": {
                "min_value": 1.5
            }
        },
        "hh_deltaeta_veto": {
            "ggF": {
                "max_value": 1.5
            }
        },
        "hh_mass_veto": {
            "ggF": {
                "max_value": 1.6
            }
        }
    }
}
```