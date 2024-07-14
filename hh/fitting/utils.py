import uproot


def convert_hists_2_root(hists, config):
    output_dir = config["General"]["InputPath"]
    for sample_name, hist_dict in hists.items():
        for hist_name, hist in hist_dict.items():
            hist_path = output_dir / f"{sample_name}_{hist_name}.root"
            with uproot.recreate(hist_path) as f:
                f["edges"] = hist["edges"]
                f["values"] = hist["values"]
                if "errors" in hist:
                    f["errors"] = hist["errors"]
    return
