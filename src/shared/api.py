import pyAMI.client
import pyAMI_atlas.api as AtlasAPI
import pathlib
import json

client = pyAMI.client.Client("atlas")
AtlasAPI.init()

_metadata_local_path = pathlib.Path(__file__).parent / "metadata.json"

with open(_metadata_local_path) as _md_file:
    _metadata_local = json.load(_md_file)


def get_metadata(datasetname_query):
    def _update_metadata_local(metadata):
        # TODO: Update with atlas api data
        metadata["kFactor"] = "1.0"
        metadata["luminosity"] = "26071.4"
        _metadata_local[datasetname_query] = metadata
        with open(_metadata_local_path, "w") as _md_file:
            json.dump(_metadata_local, _md_file)

    if _metadata_local.get(datasetname_query):
        return _metadata_local[datasetname_query]
    else:
        datasets = []
        try:
            datasets = AtlasAPI.list_datasets(
                client,
                datasetname_query,  # "%801168.Py8EG_A14NNPDF23LO_jj_JZ3.%e8453_s3873_r13829_p5440"
                type="DAOD_PHYS",
            )
        except Exception as e:
            raise Exception(
                f"Could not connect to AMI. Try running 'voms-proxy-init -voms atlas' and try again. Error: {e}"
            )
        if not datasets:
            raise Exception(
                f'Could not deduce dataset name from input path. AMI was queried with term: "{datasetname_query}"'
            )
        dataset_name = datasets[0]["ldn"]
        ds_info = AtlasAPI.get_dataset_info(
            client,
            dataset=dataset_name,
        )
        metadata = dict(ds_info[0])
        _update_metadata_local(metadata)
        return metadata
