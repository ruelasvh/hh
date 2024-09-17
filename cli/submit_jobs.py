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

import os
import json
import argparse
import htcondor
from htcondor import dags
from pathlib import Path
import shutil
from hh.shared.utils import (
    resolve_project_paths,
    concatenate_cutbookkeepers,
    get_sample_weight,
)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        type=Path,
        help="Apptainer image file to run the executable",
    )
    parser.add_argument(
        "exec",
        type=Path,
        help="Executable for the job",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Config file for the executable",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file name postfix (default: %(default)s)",
        default=Path("output.h5"),
        metavar="",
    )
    parser.add_argument(
        "-m",
        "--memory",
        type=str,
        help="Memory request for the job (default: %(default)s)",
        default="1 GB",
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
    args, restargs = get_args()

    build_response = input("Do you want to rebuild the apptainer image? (Y/n) ")
    if build_response == "" or build_response.lower() == "y":
        # get the path to the repo
        repo_path = Path(__file__).resolve().parent.parent
        os.chdir(repo_path)
        image_path = Path(os.getenv("APPTAINER_CACHEDIR")) / args.image.name
        os.system(f"apptainer build -F {image_path} Apptainer.def")
        os.chdir(os.getenv("PWD"))

    # create directories for to store files
    CWD = Path(os.getenv("PWD"))
    CONFIG_DIR = CWD / "configs"
    OUTPUT_DIR = CWD / "output"
    CONDOR_DIR = CWD / "condor"
    CONDOR_OUTPUT_DIR = CONDOR_DIR / "output"
    CONDOR_ERROR_DIR = CONDOR_DIR / "error"
    CONDOR_LOG_DIR = CONDOR_DIR / "log"
    for d in [
        CONFIG_DIR,
        OUTPUT_DIR,
        CONDOR_DIR,
        CONDOR_OUTPUT_DIR,
        CONDOR_ERROR_DIR,
        CONDOR_LOG_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

    with open(args.config) as f:
        config = resolve_project_paths(json.load(f))

    # Create a config file for each file in 'paths' for each sample in 'samples'
    configs = []
    for sample in config["samples"]:
        sample_is_mc = "data" not in sample["label"]
        for sample_path, sample_metadata, i_sample in zip(
            sample["paths"], sample["metadata"], range(len(sample["paths"]))
        ):
            sample_weight = 1.0
            if sample_is_mc:
                # get the cutbookkeepers for the sample
                cbk = concatenate_cutbookkeepers(sample_path)
                sample_weight = get_sample_weight(sample_metadata, cbk)
            # iterate over each file in the sample path for faster processing
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
                                    "metadata": [sample_metadata],
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
    hh_sub = htcondor.Submit(
        {
            "executable": "/usr/bin/apptainer",
            "arguments": f"exec {args.image} {args.exec} $(config_file) -o {OUTPUT_DIR}/{args.output.stem}_$(ClusterId)_$(ProcId){args.output.suffix} -w $(sample_weight) -v {' '.join(restargs)}",
            "output": f"{CONDOR_OUTPUT_DIR}/{args.exec.stem}-$(ClusterId).$(ProcId).log",
            "error": f"{CONDOR_ERROR_DIR}/{args.exec.stem}-$(ClusterId).$(ProcId).log",
            "log": f"{CONDOR_LOG_DIR}/{args.exec.stem}-$(ClusterId).$(ProcId).log",
            "request_memory": args.memory,
            "request_cpus": args.cpus,
            "should_transfer_files": "no",
            "my.sendcredential": "true",
            "getenv": "false",
        }
    )

    dag = dags.Dag()

    main_layer = dag.layer(
        name="hh_cmd",
        submit_description=hh_sub,
        vars=configs,
    )

    merge_sub = htcondor.Submit(
        {
            "executable": "/usr/bin/apptainer",
            "arguments": f"exec hh4b_merge {OUTPUT_DIR} -o {args.output.stem}_$(ClusterId)_$(ProcId)_merged.parquet}",
            "output": f"{CONDOR_OUTPUT_DIR}/merge-$(ClusterId).$(ProcId).log",
            "error": f"{CONDOR_ERROR_DIR}/merge-$(ClusterId).$(ProcId).log",
            "log": f"{CONDOR_LOG_DIR}/merge-$(ClusterId).$(ProcId).log",
            "request_memory": args.memory,
            "request_cpus": args.cpus,
            "should_transfer_files": "no",
            "my.sendcredential": "true",
            "getenv": "false",
        }
    )

    merge_layer = dag.layer(
        name="merge_cmd",
        submit_description=merge_sub,
    )

    # write the DAG to disk
    dag_dir = (CWD / 'hh-dag').absolute()

    # blow away any old files
    shutil.rmtree(dag_dir, ignore_errors=True)

    # make the magic happen!
    dag_file = dags.write_dag(dag, dag_dir)

    # submit the DAG
    dag_submit = htcondor.Submit.from_dag(str(dag_file), {'force': 1})

    print(dag_submit)

    # now we can enter the DAG directory and submit the DAGMan job, which will execute the graph
    os.chdir(dag_dir)
    schedd = htcondor.Schedd()
    submit_result = schedd.submit(dag_submit)

    dag_job_log = f"{dag_file}.dagman.log"
    print(f"DAG job log file is {dag_job_log}")

if __name__ == "__main__":
    main()
