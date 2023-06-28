"""
Functions from Lucas for applying the trigger buckets.

Taken from this notebook:
https://gitlab.cern.ch/lborgna/hh4b-buckets/-/blob/master/Notebooks/Assigning%20Buckets%20with%20PAG%20NNT.ipynb

Modified for easyjet Ntuples
Mar 2023
"""

import numpy as np
import pandas as pd
import awkward as ak
import operator as SmoothOperator

from .utils import triggers_NR as triggers


def get_nr_bucket(t, evts, mask, year, jalias):
    """
    Get the triggers for the year and the NR bucket.
    """

    jpts = t.arrays("pt", aliases=jalias).pt[mask]
    jpts_sort = ak.sort(jpts, ascending=False)

    evts["lead_pt"] = jpts_sort[:, 0].to_numpy()
    evts["third_pt"] = jpts_sort[:, 2].to_numpy()

    # num_jets = ak.count(jpts_sort,axis=1)
    # evts['third_pt'] = -1
    # mask = (num_jets > 3).to_numpy()
    # evts.loc[mask,'third_pt'] = jpts_sort[mask, 2].to_numpy()

    tDict = trigger_lut(year)

    # Trigger matching -- still needs to be implemented in easyjet fw
    # for ti in triggers[year]:
    #     trigHashes = t['matchedTriggerHashes'].array()
    #     evts[ti] = ak.any(trigHashes==hashMap[ti],axis=1)

    # trig_cols = [f'triggerPassed_{ti}' for ti in triggers[year]]
    trig_cols = triggers[year]

    evts["trigger"] = evts[trig_cols].sum(axis=1).astype(bool)
    evts["nr_bucket"] = 0

    b1_mask = (evts["lead_pt"] > 170) & (evts["third_pt"] > 70)

    evts.loc[b1_mask & evts[tDict["2b1j"]].values, "nr_bucket"] = 1
    evts.loc[~b1_mask & evts[tDict["2b2j"]].values, "nr_bucket"] = 2


def trigger_lut(year: int) -> dict:
    """
    Lookup table for trigger keys for each year
    """
    if (year == 2016) or (year == 16):
        return {
            "1b": "HLT_j225_bmv2c2060_split",
            "2b1j": "HLT_j100_2j55_bmv2c2060_split",
            "2bHT": "HLT_2j35_bmv2c2060_split_2j35_L14J15p0ETA25",
            "2b2j": "HLT_2j35_bmv2c2060_split_2j35_L14J15p0ETA25",
        }
    elif (year == 2017) or (year == 17):
        return {
            "1b": "HLT_j225_gsc300_bmv2c1070_split",
            "2b1j": "HLT_j110_gsc150_boffperf_split_2j35_gsc55_bmv2c1070_split_L1J85_3J30",
            "2bHT": "HLT_2j35_gsc55_bmv2c1050_split_ht300_L1HT190_J15s5pETA21",
            "2b2j": "HLT_2j15_gsc35_bmv2c1040_split_2j15_gsc35_boffperf_split_L14J15p0ETA25",
        }
    elif (year == 2018) or (year == 18):
        return {
            "1b": "HLT_j225_gsc300_bmv2c1070_split",
            "2b1j": "HLT_j110_gsc150_boffperf_split_2j45_gsc55_bmv2c1070_split_L1J85_3J30",
            "2bHT": "HLT_2j45_gsc55_bmv2c1050_split_ht300_L1HT190_J15s5pETA21",
            "2b2j": "HLT_2j35_bmv2c1060_split_2j35_L14J15p0ETA25",
        }


def bucket_config_generator(
    cut_bucket1: float = 325,
    cut_bucket2: float = 170,
    cut_bucket3: float = 900,
    t_1b: str = "HLT_j225_gsc300_bmv2c1070_split",
    t_2b1j: str = "HLT_j110_gsc150_boffperf_split_2j35_gsc55_bmv2c1070_split_L1J85_3J30",
    t_2bHT: str = "HLT_2j35_gsc55_bmv2c1050_split_ht300_L1HT190-J15s5.ETA21",
    t_2b2j: str = "HLT_2j15_gsc35_bmv2c1040_split_2j15_gsc35_boffperf_split_L14J15.0ETA25",
):
    """
    Returns a dictionary of the configuration for the nominal bucketing scheme.
    note: variable names have been renamed to support Nicole's frameworks.
    """
    BucketsDict = {
        "Bucket1": {
            "offVar": ["lead_pt", "lead_tag"],
            "offVarCut": [cut_bucket1, 1],
            "operator": [SmoothOperator.gt, SmoothOperator.eq],
            "trigger": t_1b,
        },
        "Bucket2": {
            "offVar": ["lead_pt", "lead_tag"],
            "offVarCut": [cut_bucket2, 0],
            "operator": [SmoothOperator.gt, SmoothOperator.eq],
            "trigger": t_2b1j,
        },
        "Bucket3": {
            "offVar": ["HT_all"],
            "offVarCut": [cut_bucket3],
            "operator": [SmoothOperator.gt],
            "trigger": t_2bHT,
        },
        "Bucket4": {
            "offVar": [],
            "offVarCut": [],
            "operator": [],
            "trigger": t_2b2j,
        },
    }

    return BucketsDict


def offline_mask(df: pd.DataFrame, bucket: dict, data_size=None) -> pd.Series:
    """Generates the offline mask for a specific bucket

    Args:
        bucket (dict): Dict containing the specifics of the buckets (offVar, offVarCut and operator)
        df (pd.DataFrame): DataFrame that the offline mask is to be applied on.
        data_size (int): size of mask to generate, only needed to reduce calculations.

    Returns:
        mask (pd.Series): boolean mask indicating if the event passed offline variable cut for bucket.
    """
    if data_size == None:
        data_size = df.shape[0]

    mask = True  # pd.Series(np.ones(data_size, dtype=bool))  # Needs to be a pd.Series
    for op, offVar, offVarCut in zip(
        bucket["operator"], bucket["offVar"], bucket["offVarCut"]
    ):
        mask = mask & op(df[offVar].values, offVarCut)
    return mask


def assign_bucket(df: pd.DataFrame, bucket_config: dict, category_name: str = "bucket"):
    """
    Assigns the bucket categorization to the dataframe with the provided configuration.

    Args:
        df (pd.DataFrame): data to apply bucketing on
        bucket_config (dict): configuration of parameters for bucketing
        category_name (str): name of bucket category column

    bucket_config should contain the configuration information for how the buckets should be assigned

    #Bucket1 offline mask = lead_jet_pT > 300 AND lead_jet_tag == 1
        #Bucket1 = Bucket1 offline mask AND Bucket1 trigger Mask

    #Bucket2 offline mask = NOT bucket1 offline mask and lead_jet_pt > 200 and lead_jet_tag == 0
        # Bucket 2 = Bucket2 offline mask AND Bucket2 trigger Mask

    #Bucket3 offline mask = NOT bucket1 offline mask and NOT bucket2 offline mask and HT > 600
        # Bucket 3 = Bucket3 offline mask AND Bucket3 trigger Mask

    #Bucket4 offline mask = NOT bucket1 offline mask and NOT bucket2 offline mask and NOT bucket 3 offline mask and  bucket4 trigger Mask == True
        #Bucket 4
    """

    data_size = df.shape[0]

    B1_offline_mask = offline_mask(df, bucket_config["Bucket1"], data_size)
    B1_trig_mask = df[bucket_config["Bucket1"]["trigger"]]
    B1_mask = B1_offline_mask & B1_trig_mask

    B2_offline_mask = offline_mask(df, bucket_config["Bucket2"], data_size)
    B2_trig_mask = df[bucket_config["Bucket2"]["trigger"]]
    B2_mask = ~B1_offline_mask & B2_offline_mask & B2_trig_mask

    B3_offline_mask = offline_mask(df, bucket_config["Bucket3"], data_size)
    B3_trig_mask = df[bucket_config["Bucket3"]["trigger"]]
    B3_mask = ~B1_offline_mask & ~B2_offline_mask & B3_offline_mask & B3_trig_mask

    B4_offline_mask = offline_mask(df, bucket_config["Bucket4"], data_size)
    B4_trig_mask = df[bucket_config["Bucket4"]["trigger"]]
    B4_mask = (
        ~B1_offline_mask
        & ~B2_offline_mask
        & ~B3_offline_mask
        & B4_offline_mask
        & B4_trig_mask
    )

    df[category_name] = 0
    df.loc[B1_mask, category_name] = 1
    df.loc[B2_mask, category_name] = 2
    df.loc[B3_mask, category_name] = 3
    df.loc[B4_mask, category_name] = 4

    return df


def get_res_bucket(t, evts, mask, year, jalias):
    """
    Get the resonant bucket
    """

    trig_yr = trigger_lut(year)

    jarr = t.arrays(
        ["pt", "tag"], aliases=jalias, cut=f"(pt > {25}) & (abs(eta) < {2.5})"
    )
    jarr = jarr[mask]

    evts["lead_pt"] = ak.max(jarr["pt"], axis=1)

    idx_lead = ak.argmax(jarr["pt"], axis=1)
    evts["lead_tag"] = jarr["tag"][np.arange(len(idx_lead)), idx_lead.to_numpy()]

    evts["HT_all"] = ak.sum(jarr["pt"], axis=1)

    bucket_config = bucket_config_generator(
        t_1b=trig_yr["1b"],
        t_2b1j=trig_yr["2b1j"],
        t_2bHT=trig_yr["2bHT"],
        t_2b2j=trig_yr["2b2j"],
    )

    evts = assign_bucket(evts, bucket_config)


def getOnlineTag(df):
    """
    Goal: Retrieve the online b-tagging decision for the jets that *failed* the
    offline 77% WP.

    To study whether we can use PC information, we wanted to investigate the
    impact of j dropping the events where the "not b" jet matched to an online b-tag.

    For each of the triggers matched and considered by XhhCommon, this function adds
    a new column: hlt_{1b,2b1j,2bHT,2b2j}_notOfflineTagOnlineDecision which is
    * Always False for 4b
    * Online decision for the "not b @ 77% offline WP" for 3b events
    * "Or" of online decsion for the 2b events

    Then we can veto these events by applying a ~ df[hlt_{1b,2b1j,2bHT,2b2j}_notOfflineTagOnlineDecision].
    """

    trig_keys = ["1b", "2b1j", "2bHT", "2b2j"]

    # Loop over and consider each of the triggers separately
    for ti in trig_keys:
        # Init the columns
        df[f"hlt_{ti}_notOfflineTagOnlineDecision"] = False

        for ntag in [3, 2]:
            mask = df.ntag == ntag

            notOfflineTag = ~df.loc[
                mask, [f"j{i}_btag" for i in range(4)]
            ].values.astype(bool)
            onlineTag = (
                df.loc[mask, [f"j{i}_sf_{ti}" for i in range(4)]]
                .values[notOfflineTag]
                .reshape(-1, 2)
            )

            df.loc[mask, f"hlt_{ti}_notOfflineTagOnlineDecision"] = np.any(
                onlineTag == 1, axis=1
            )
