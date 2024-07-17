import uproot
import numpy as np


def convert_hists_2_root(hists, config):
    hist_path = config["General"]["InputPath"].split(":")[0]
    with uproot.recreate(hist_path) as f:
        for region in config["Regions"]:
            hist_edges = np.array(region["Binning"])
            for sample in config["Samples"]:
                if sample.get("Data", False):
                    f[
                        f"{region['RegionPath']}/{sample['SamplePath']}/{config['General']['VariationPath']}"
                    ] = (np.zeros(len(hist_edges) - 1), hist_edges)
                else:
                    sample_hist_name = sample["Name"].split("/")
                    f[
                        f"{region['RegionPath']}/{sample['SamplePath']}/{config['General']['VariationPath']}"
                    ] = (
                        hists[sample_hist_name[0]][sample_hist_name[1]]["values"][
                            1:-1
                        ],  # Remove underflow and overflow bins
                        hist_edges,
                    )

    return
