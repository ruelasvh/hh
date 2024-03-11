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
from hh.shared.utils import (
    resolve_project_paths,
    concatenate_cutbookkeepers,
    get_sample_weight,
)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "executable",
        type=Path,
        help="Executable for the job",
    )
    parser.add_argument("config", type=Path, help="Config file for the executable")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file name postfix (default: %(default)s)",
        default=Path("output"),
        metavar="",
    )
    parser.add_argument(
        "-m",
        "--memory",
        type=str,
        help="Memory request for the job (default: %(default)s)",
        default="5 GB",
        metavar="",
    )
    parser.add_argument(
        "-c",
        "--cpus",
        type=int,
        help="Number of cpus for the job (default: %(default)s)",
        default=1,
        metavar="",
    )

    return parser.parse_known_args()


def main():
    CONFIG_DIR = Path("/lustre/fs22/group/atlas/ruelasv/tmp/condor/configs")
    OUTPUT_DIR = Path("/lustre/fs22/group/atlas/ruelasv/tmp/condor/output")
    ERROR_DIR = Path("/lustre/fs22/group/atlas/ruelasv/tmp/condor/error")
    LOG_DIR = Path("/lustre/fs22/group/atlas/ruelasv/tmp/condor/log")

    args, restargs = get_args()
    with open(args.config) as f:
        config = resolve_project_paths(json.load(f))

    # Create a config file for each file in 'paths' for each sample in 'samples'
    configs = []
    for sample in config["samples"]:
        sample_is_mc = "data" not in sample["label"]
        for i, sample_path in enumerate(sample["paths"]):
            sample_weight = 1.0
            if sample_is_mc:
                # get the cutbookkeepers for the sample
                cbk = concatenate_cutbookkeepers(sample_path)
                sample_weight = get_sample_weight(sample["metadata"][i], cbk)
            # iterate over each file and create a config file for it as described in the docstring
            files = list(Path(sample_path).glob("*.root"))
            for file_path in files:
                # create a config file for each file
                config_file = (
                    CONFIG_DIR / f"config-{sample['label']}-{file_path.stem}.json"
                )
                # add the config file to the list of configs
                configs.append(
                    {
                        "config_file": config_file.as_posix(),
                        "sample_weight": str(sample_weight),
                    }
                )
                # create a new config file for each file
                with open(config_file, "w") as file:
                    # write the sample and event_selection objects to the new config file
                    json.dump(
                        {
                            **config,
                            "samples": [
                                {
                                    **sample,
                                    "paths": [
                                        (file_path.parent / file_path.stem).as_posix()
                                    ],
                                }
                            ],
                        },
                        file,
                        indent=4,
                    )

    ########################################################
    # Submit a job for each config file
    ########################################################

    # token management needed for submitting jobs to HTCondor in python
    credd = htcondor.Credd()
    credd.add_user_cred(htcondor.CredTypes.Kerberos, None)

    # create a submit object using htcondor.Submit
    sub = htcondor.Submit(
        {
            "executable": args.executable,
            "should_transfer_files": "no",
            "my.sendcredential": "true",
            "arguments": f"$(config_file) -o {args.output.stem}_$(ClusterId)_$(ProcId) -w $(sample_weight) -v {' '.join(restargs)}",
            "output": f"{OUTPUT_DIR}/{args.executable.stem}-$(ClusterId).$(ProcId).out",
            "error": f"{ERROR_DIR}/{args.executable.stem}-$(ClusterId).$(ProcId).err",
            "log": f"{LOG_DIR}/{args.executable.stem}-$(ClusterId).$(ProcId).log",
            "request_memory": args.memory,
            "request_cpus": args.cpus,
            "getenv": "PYTHONPATH",
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
