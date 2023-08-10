# Plot HH4b Analysis

## Installation
```bash
git clone https://github.com/ruelasvh/plot-hh4b-analysis.git
cd plot-hh4b-analysis
python3 -m pip install -e .
```

## Usage
```bash
hh4b_non_res_res_make_hists config.json
```

For example, have a look in [this config file](src/nonresonantresolved/configs/config-test.json).

The output will be an `h5` file. To make the actual plots, in the root directory of the project run:

```bash
hh4b_non_res_res_draw_hists hists.h5
```