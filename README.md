# Tools for HH4b Analysis

## Installation
```bash
git clone https://github.com/ruelasvh/hh.git
cd hh
python3 -m pip install -e .
```

## Creating histograms and making plots
```bash
hh4b_non_res_res_make_hists config.json -v
```

For example, have a look in [this config file](src/nonresonantresolved/configs/config-test.json).

The output will be an `h5` file. To make the actual plots, in the root directory of the project run:

```bash
hh4b_non_res_res_draw_hists hists.h5
```

## Dumping information for ML studies
```bash
hh4b_dump config.json -v
```

This will create a `.h5` file with the information specifiend in `config.json`. If you want to instead dump the information into a `root` file, then you can do:

```bash
hh4b_dump config.json -v --output output.root
```

## Running jobs in parallel (htcondor)
First create an environment variable where the logs for the jobs will be stored:

```bash
export HH4B_LOGS_DIR=/path/to/logs
```

Then, submit the jobs with:
```bash
hh4b_submit [hh4b_non_res_res_make_hists|hh4b_dump] config.json -v
```

The output will be stored to the working directory.

For any of the commands, you can use the `-h` flag to get more information on the options available.
