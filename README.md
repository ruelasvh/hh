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

For example, have a look in [this config file](src/nonresonantresolved/config-test.json).

The `input` paths need to be nested in a folder with the name of the project (e.g. `mc21_13p6TeV.hh4b.ggF`) and the sample name (e.g. `user.viruelas.HH4b.2023_04_21.601479.PhPy8EG_HH4b_cHHH01d0.e8472_s3873_r13829_p5631_TREE`) to be able to fetch metadata from AMI. So the full path to the input files would be `path/to/k01inputsdir/mc21_13p6TeV.hh4b.ggF/user.viruelas.HH4b.2023_04_21.601479.PhPy8EG_HH4b_cHHH01d0.e8472_s3873_r13829_p5631_TREE/`.

The output will be an `h5` file. To make the actual plots, in the root directory of the project run:

```bash
hh4b_non_res_res_draw_hists hists.h5
```