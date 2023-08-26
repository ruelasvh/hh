#!/usr/bin/env python3

"""
Submits jobs to HTCondor.
It takes a config file, such as src/nonresonantresolved/configs/config-test.json,
goes over each sample in 'samples'. For each sample, it checks 'paths' for files
and creates a seperate config-sample-fileID.json (saves it in tmp/configs/) 
file for each file for that sample only and removes the other samples. It also 
keeps the object 'event_selection'. For each config-sample-fileID.json it submits 
a job to HTCondor.
"""

import json
import argparse
import htcondor
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="Output file name postfix",
    )
    parser.add_argument(
        "-e",
        "--executable",
        type=Path,
        default=Path(
            "/afs/ifh.de/user/r/ruelasv/.local/bin/hh4b_non_res_res_make_hists"
        ),
        help="Executable (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    CONFIG_DIR = Path("/lustre/fs22/group/atlas/ruelasv/tmp/condor/configs")
    OUTPUT_DIR = Path("/lustre/fs22/group/atlas/ruelasv/tmp/condor/output")
    ERROR_DIR = Path("/lustre/fs22/group/atlas/ruelasv/tmp/condor/error")
    LOG_DIR = Path("/lustre/fs22/group/atlas/ruelasv/tmp/condor/log")

    args = get_args()
    with open(args.config) as f:
        config = json.load(f)

    # Create a config file for each file in 'paths' for each sample in 'samples'
    configs = []
    for sample in config["samples"]:
        for path in sample["paths"]:
            # 'path' is a directory, so get all files in that directory
            # if Path(path).is_dir():
            if path:
                # remove .root ext from files
                files = [f.parent / f.stem for f in Path(path).glob("*.root")]
                # iterate over each file and create a config file for it as described in the docstring
                for file in files:
                    # create a config file for each file
                    config_file = (
                        CONFIG_DIR / f"config-{sample['label']}-{file.stem}.json"
                    )
                    # add the config file to the list of configs
                    configs.append({"config_file": config_file.as_posix()})
                    # create a new config file for each file
                    with open(config_file, "w") as f:
                        # write the sample and event_selection objects to the new config file
                        json.dump(
                            {
                                **config,
                                "samples": [{**sample, "paths": [file.as_posix()]}],
                            },
                            f,
                            indent=4,
                        )

    ########################################################
    # Submit a job for each config file
    ########################################################

    # token management needed for submitting jobs to HTCondor in python
    # col = htcondor.Collector()
    credd = htcondor.Credd()
    credd.add_user_cred(htcondor.CredTypes.Kerberos, None)

    # create a submit object using htcondor.Submit
    sub = htcondor.Submit(
        {
            "executable": f"{args.executable}",
            "should_transfer_files": "no",
            "my.sendcredential": "true",
            "arguments": f"$(config_file) -o {args.output.stem}_$(ClusterId)_$(ProcId) -j 2 -v",
            "output": f"{OUTPUT_DIR}/{args.executable.stem}-$(ClusterId).$(ProcId).out",
            "error": f"{ERROR_DIR}/{args.executable.stem}-$(ClusterId).$(ProcId).err",
            "log": f"{LOG_DIR}/{args.executable.stem}-$(ClusterId).$(ProcId).log",
            "request_memory": "6 GB",
            "request_cpus": "2",
            "request_disk": "2 GB",
        }
    )

    # create a schedd object using htcondor.Schedd
    schedd = htcondor.Schedd()
    # submit one job for each item in the configs
    submit_result = schedd.submit(sub, itemdata=iter(configs))

    with open(f"submitted-jobs-{submit_result.cluster()}.log", "w") as f:
        # write the cluster id and number of processes to a file
        f.write(f"ClusterID: {submit_result.cluster()}\n")
        f.write(f"Number of submitted jobs: {submit_result.num_procs()}\n")


if __name__ == "__main__":
    main()
