CAMPAIGNS = {
    2016: ["r13167", "r14859"],
    2017: ["r13144", "r14860"],
    2018: ["r13145", "r14861"],
    2022: ["r14622", "r14932"],
    2023: ["r15224"],
}


def get_trigger_branch_aliases(selections, sample_year=None):
    trig_aliases = {}
    for selection_config in selections.values():
        if "trigs" in selection_config and "value" in selection_config["trigs"]:
            if sample_year in selection_config["trigs"]["value"]:
                trig_aliases.update(
                    {
                        f"trig_{trig_short}": f"trigPassed_{trig_long}"
                        for trig_short, trig_long in selection_config["trigs"]["value"][
                            sample_year
                        ].items()
                    }
                )
    return trig_aliases
