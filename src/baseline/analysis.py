"""
analyais.py

Goal: Given one of the multivariate (or baseline) pairing algs, this code just does evaluation
of looping over the jets, forming the pairs, and defining columns for the other analysis cut
(deta_hh and Xwt) and the kinematic regions

Updated
Mar 2023
"""

import numpy as np
import pandas as pd

import uproot
import awkward as ak
import vector

# from coffea.nanoevents.methods import vector

# ak.behavior.update(vector.behavior)

from .trigger import get_nr_bucket, get_res_bucket, trigger_lut
from .truth import truthMatchJets, getCorrectPair

# import pyarrow as pa
# import pyarrow.parquet as pq

vecDict = {"pt": "pt", "eta": "eta", "phi": "phi", "mass": "m"}
jvtStr = "(pt > 60) | ( (pt>20.00000586) & (pt<24.08409573) & (eta>-2.4999998) & (eta<-1.5580761) & (NNJvt>0.27)) | ( (pt>20.00000586) & (pt<24.08409573) & (eta>-1.5580761) & (eta<-0.50954574) & (NNJvt>0.74)) | ( (pt>20.00000586) & (pt<24.08409573) & (eta>-0.50954574) & (eta<0.51835537) & (NNJvt>0.74)) | ( (pt>20.00000586) & (pt<24.08409573) & (eta>0.51835537) & (eta<1.5639215) & (NNJvt>0.74)) | ( (pt>20.00000586) & (pt<24.08409573) & (eta>1.5639215) & (eta<2.4999998) & (NNJvt>0.27)) | ( (pt>24.08409573) & (pt<29.8005357) & (eta>-2.4999998) & (eta<-1.5580761) & (NNJvt>0.21)) | ( (pt>24.08409573) & (pt<29.8005357) & (eta>-1.5580761) & (eta<-0.50954574) & (NNJvt>0.73)) | ( (pt>24.08409573) & (pt<29.8005357) & (eta>-0.50954574) & (eta<0.51835537) & (NNJvt>0.73)) | ( (pt>24.08409573) & (pt<29.8005357) & (eta>0.51835537) & (eta<1.5639215) & (NNJvt>0.73)) | ( (pt>24.08409573) & (pt<29.8005357) & (eta>1.5639215) & (eta<2.4999998) & (NNJvt>0.19)) | ( (pt>29.8005357) & (pt<39.46663572) & (eta>-2.4999998) & (eta<-1.5580761) & (NNJvt>0.11)) | ( (pt>29.8005357) & (pt<39.46663572) & (eta>-1.5580761) & (eta<-0.50954574) & (NNJvt>0.65)) | ( (pt>29.8005357) & (pt<39.46663572) & (eta>-0.50954574) & (eta<0.51835537) & (NNJvt>0.6900000000000001)) | ( (pt>29.8005357) & (pt<39.46663572) & (eta>0.51835537) & (eta<1.5639215) & (NNJvt>0.66)) | ( (pt>29.8005357) & (pt<39.46663572) & (eta>1.5639215) & (eta<2.4999998) & (NNJvt>0.1)) | ( (pt>39.46663572) & (pt<60) & (eta>-2.4999998) & (eta<-1.5580761) & (NNJvt>0.07)) | ( (pt>39.46663572) & (pt<60) & (eta>-1.5580761) & (eta<-0.50954574) & (NNJvt>0.63)) | ( (pt>39.46663572) & (pt<60) & (eta>-0.50954574) & (eta<0.51835537) & (NNJvt>0.75)) | ( (pt>39.46663572) & (pt<60) & (eta>0.51835537) & (eta<1.5639215) & (NNJvt>0.51)) | ( (pt>39.46663572) & (pt<60) & (eta>1.5639215) & (eta<2.4999998) & (NNJvt>0.07))"
mH = 125  # GeV


def processDf(
    fname,
    tname="AnalysisMiniTree",
    nJetsMax=4,
    year=2016,
    pT_min=40,
    eta_max=2.5,
    tagger="DL1d",
    WP=77,
    sort="tag",
    min_btags=3,
    mc=True,
    truth=True,
    bjetTrigSFs=False,
    truthJets=False,
):
    """
    Given an input MNT, get the relevant jet features and apply the ml preprocessing

    Inputs:
    - filename: Name of the mnt file to read in the name for
    - treename: The MNT tree name to read in
    - nJetsMax: Max # of jets to consider
    - year: Year to calculate the corresponding triggers for (default 2016)
    - pT_min: Min jet pT to consider
    - eta_max: eta cut
    - sort: The sort to apply to the jets before truncating.
            Options: tag (default), pt (default), etc
    - min_btags: How many b-tags to save
    - mc: Flag for whether filename is mc
    - truth: flag for if truth info is stored in the MNT

    Returns:
    - df: Pandas df w/ the relevant jet and event level info calculated from the MNT.
    """

    # Open the file
    t = uproot.open(f"{fname}:{tname}")

    # Get the triggers corresponding to this year
    trig_yr = trigger_lut(year)
    trig_cols = trig_yr.values()

    # Load in the corresponding arrays
    ecols = ["eventNumber", "runNumber"]
    if mc:
        # ecols += ["pileupWeight", "generatorWeight"]
        ecols += ["pileupWeight"]
    ealias = {
        "pileupWeight": "PileupWeight_NOSYS",
        # "generatorWeight": "generatorWeight_NOSYS",
    }

    # ecols += trig_cols
    # for trig in trig_cols:
    #     ealias[trig] = f"trigPassed_{trig}"

    evts = t.arrays(ecols, aliases=ealias, library="pd")
    print(f"Starting events = {len(evts.index)}")

    # Load in the mc event weight
    if mc:
        arr = t.arrays("mcEventWeights")
        evts["mcEventWeight"] = arr["mcEventWeights"][:, 0]
        del arr

    print("Loading in the jet array")

    jcols = ["pt", "eta", "phi", "m", "tag", "NNJvt"]
    jcols_full = [
        f"recojet_antikt4_NOSYS_{v}"
        for v in [
            "pt",
            "eta",
            "phi",
            "m",
            f"ftag_select_{tagger}v01_FixedCutBEff_{WP}",
            "NNJvt",
        ]
    ]
    if mc:
        jcols += ["flav"]
        jcols_full += ["recojet_antikt4_NOSYS_HadronConeExclTruthLabelID"]

    jalias = {
        k: f"{v} / 1000" if (k == "pt" or k == "m") else v
        for k, v in zip(jcols, jcols_full)
    }
    # jalias['sf'] = f'recojet_antikt4_NOSYS_SF_{tagger}_FixedCutBEff_{WP}'

    jarr = t.arrays(
        jcols,
        aliases=jalias,
        cut=f"(pt > {pT_min}) & (abs(eta) < {eta_max}) & ({jvtStr})",
    )

    # Mask events passing the trigger and 4-jet selections
    evts["njets"] = ak.num(jarr["pt"]).to_numpy()
    evts["ntag"] = ak.sum(jarr["tag"], axis=1).to_numpy()
    four_jets = (evts["njets"] >= 4) & (evts["ntag"] >= min_btags)
    print(f"Cutting on 4 jets and {min_btags} b-tags: ", four_jets.sum(), "evts")

    if four_jets.sum() == 0:
        return None, None

    jarr = jarr[four_jets]
    evts = evts[four_jets.to_numpy()].reset_index(drop=True)

    # # Get the analysis trigger decision
    # print(f"Applying {year} triggers")

    # get_nr_bucket(t, evts, four_jets, year, jalias)
    # get_res_bucket(t, evts, four_jets, year, jalias)

    # print("Using the resonant  buckets")
    # mask = evts.bucket > 0
    # jarr = jarr[mask]
    # evts = evts[mask].reset_index(drop=True)

    """
    Apply the jet selections
    """
    # First sort by the jet pt
    idx0 = ak.argsort(jarr["pt"], ascending=False)
    jarr = jarr[idx0]

    # Now sort by whatever else (default btag decision)
    idx = ak.argsort(jarr[sort], ascending=False)

    # Use the central njets (w/o the vbf included) to make X_wt
    print("Calculating X_wt")
    j4 = vector.zip(
        {k: jarr[v] for k, v in vecDict.items()},
    )
    evts["X_wt_tag"] = getXwt(jarr, j4, idx).to_numpy()

    hc_jets = j4[idx[:, :nJetsMax]]

    if mc:
        # The ftag sfs use the same jet cuts as the central jets
        print("Calculating the SF")
        # jsfs = t.arrays('sf', aliases=jalias, cut=f'(pt > {pT_min}) & (abs(eta) < {eta_max}) & {jvtCut}')
        # mc_sf = ak.prod(jsfs.sf[:,:,0],axis=-1).to_numpy()
        mc_sf = np.prod(evts[["mcEventWeight", "pileupWeight"]], axis=1)

        evts["mc_sf"] = mc_sf

        # Also save the truth info
        if truth:
            print("Retrieving the truth information")

            # Get the b-quark array and 4-vectors
            tkeys = [k for k in t.keys() if "truth_" in k]
            taliases = {
                k[6:]: f"{k}/1000" if ((k[-2:] == "pt") or (k[-1] == "m")) else k
                for k in tkeys
            }
            tarr = t.arrays(taliases.keys(), aliases=taliases)

            tarr = tarr[four_jets][mask]

            H_bs = vector.zip(
                {k: tarr[f"bb_fromH1_{v}"] for k, v in vecDict.items()},
            )
            S_bs = vector.zip(
                {k: tarr[f"bb_fromH2_{v}"] for k, v in vecDict.items()},
            )

            b4 = ak.concatenate([H_bs, S_bs], 1)

            truthMatchJets(evts, hc_jets, b4, nJetsMax)
            getCorrectPair(evts, nJetsMax)

        if truthJets:
            raise NotImplementedError
            # getTruthJets(tree, evts, nJetsMax)

    return evts, hc_jets


def getXhh(m1, m2, c1=125, c2=125, res1=0.1, res2=0.1):
    """
    Return the Xhh value for the event

    Inputs:
    - m1, m2: The 4-vectors for the leading and subleading HCs in the event
    - c1, c2: The center for the ellipse in the (lead,subl) HC mass plane
    - mres: The radius of the elipse, which will be multiplied by the HC mass

    Output:
    - Xhh: The Xhh value for the event, eq (3) from the 36 ifb int note
    """
    return np.sqrt(((m1 - c1) / (res1 * m1)) ** 2 + ((m2 - c2) / (res2 * m2)) ** 2)


def getXwt(jarr, j4, idx):
    """
    Calculate the X_wt variable for the event.

    Note: consistent w/ RR, only considers the HC jets (the first 4 jets in idx) for the
    b-tagged b-jets in the top-candidate.

    Input:
    - jarr: awkward array of jet features
    - ps: awkward array of 4-vectors for the jets
    - idx: ordering for the jets (first 4 jets are the HC jets in the eveny)

    Output:
    - Xwt: The Xwt minimized over all of the valid 3-jet combinations
    """
    ps = j4[idx]
    btag = jarr.tag[idx]

    bjet = ps[:, :4][btag[:, :4] == 1]
    bidx = ak.Array([range(nb) for nb in ak.num(bjet)])

    # # add a dim across the last entry
    bjet = bjet[:, :, np.newaxis]
    bidx = bidx[:, :, np.newaxis]

    w_jet_pairs = ak.combinations(ps, 2)
    w_idx_pairs = ak.argcombinations(ps, 2)

    wjet1, wjet2 = ak.unzip(w_jet_pairs[:, np.newaxis, :])
    widx1, widx2 = ak.unzip(w_idx_pairs[:, np.newaxis, :])

    # Get the corresponding combinations
    m_W = 80.4
    m_t = 172.5

    WC = wjet1 + wjet2
    tC = bjet + WC

    Xwt_combs = getXhh(tC.mass, WC.mass, m_t, m_W)

    # Set as "invalid" the entries where the b-jet overlaps w/ one of the w-jets
    Xwt_mask = ak.where((bidx == widx1) | (bidx == widx2), np.inf, Xwt_combs)
    Xwt_discrim_min = ak.min(ak.min(Xwt_mask, axis=-1), axis=-1)
    # print sum of events passing X_wt_tag > 1.5
    print("Events passing X_wt_tag > 1.5: ", ak.sum(Xwt_discrim_min > 1.5))
    # Minimize over the possible combinations + return
    return Xwt_discrim_min


"""
Some pairing functions
"""


def dr_lead(js, pair):
    """
    Return the delta R of the leading HC
    """
    pair_to_idx = {0: ((0, 1), (2, 3)), 1: ((0, 2), (1, 3)), 2: ((0, 3), (1, 2))}

    (ia0, ia1), (ib0, ib1) = pair_to_idx[pair]

    # pt ordering
    hc_sort = (js[:, ia0] + js[:, ia1]).pt > (js[:, ib0] + js[:, ib1]).pt

    dr = np.where(hc_sort, js[:, ia0].deltaR(js[:, ia1]), js[:, ib0].deltaR(js[:, ib1]))

    return dr.to_numpy()


def do_min_dR(evts, hc_jets):
    """
    Goal: Do the baseline analysis selection.

    Inputs:
    - evts
    - hc_jets
    """

    evts["chosenPair"] = np.argmin(
        np.vstack([dr_lead(hc_jets, i) for i in range(3)]), axis=0
    )

    ja0 = hc_jets[:, 0]
    print("chosen pair", np.unique(evts["chosenPair"].values.astype(int)))

    ja1 = hc_jets[range(len(hc_jets)), evts["chosenPair"].values.astype(int) + 1]

    jb0 = ak.where(evts["chosenPair"] == 0, hc_jets[:, 2], hc_jets[:, 1])
    jb1 = ak.where(evts["chosenPair"] == 2, hc_jets[:, 2], hc_jets[:, 3])

    hca = ja0 + ja1
    hcb = jb0 + jb1

    # The scalar candidate with the mass closest to the Higgs mass
    # will be the Higgs candidate
    sort_mask = abs(hca.mass - mH) < abs(hcb.mass - mH)
    HC = ak.where(sort_mask, hca, hcb)
    SC = ak.where(sort_mask, hcb, hca)

    # There are more possibilities for pairing in SH events (6 possibile pairings instead of 3).
    # In truth.py, correctPair = {0,1,2} if j0 is in the Higgs candidate,
    # and {3,4,5} if j0 is in the S candidate
    evts.loc[~sort_mask.to_numpy(), "chosenPair"] += 3
    print(
        "chosen pair: extended labelling",
        np.unique(evts["chosenPair"].values.astype(int), return_counts=True),
    )

    return HC, SC


def min_dH(js, pair, pair_to_idx):
    """
    Return the delta R of the leading HC
    """

    (ia0, ia1), (ib0, ib1) = pair_to_idx[pair]
    dm = abs((js[:, ia0] + js[:, ia1]).mass - mH)

    return dm.to_numpy()


def do_min_dH(evts, hc_jets):
    """
    Goal: Do the baseline analysis selection.

    Inputs:
    - evts
    - hc_jets
    """

    # Recall: This is how the possible truth pairings are defined:
    pair_to_idx = {
        0: ((0, 1), (2, 3)),
        1: ((0, 2), (1, 3)),
        2: ((0, 3), (1, 2)),
        3: ((2, 3), (0, 1)),
        4: ((1, 3), (0, 2)),
        5: ((1, 2), (0, 3)),
    }

    # The way we set this up.. the first tuple corresponds to
    # the Higgs candidate
    arr_pair_to_idx = np.array(
        [
            [0, 1, 2, 3],
            [0, 2, 1, 3],
            [0, 3, 1, 2],
            [2, 3, 0, 1],
            [1, 3, 0, 2],
            [1, 2, 0, 3],
        ]
    )

    dms = np.vstack([min_dH(hc_jets, i, pair_to_idx) for i in range(6)])
    evts["chosenPair"] = np.argmin(dms, axis=0)

    # In some sense, the pairing is easier here... b/c the pairing is the same
    # as the resonance assignment... and this is reflected with more succinct code
    # than in  do_min_dR
    r = np.arange(len(hc_jets))
    idx = arr_pair_to_idx[evts.chosenPair]

    jh0 = hc_jets[r, idx[:, 0]]
    jh1 = hc_jets[r, idx[:, 1]]

    js0 = hc_jets[r, idx[:, 2]]
    js1 = hc_jets[r, idx[:, 3]]

    # Reconstruct the resonances
    HC = jh0 + jh1
    SC = js0 + js1

    return HC, SC


def do_min_dR_all(evts, hc_jets):
    """
    Goal: Do the baseline analysis selection.

    Inputs:
    - evts
    - hc_jets
    """

    # Recall: This is how the possible truth pairings are defined:
    pair_to_idx = {
        0: ((0, 1), (2, 3)),
        1: ((0, 2), (1, 3)),
        2: ((0, 3), (1, 2)),
        3: ((2, 3), (0, 1)),
        4: ((1, 3), (0, 2)),
        5: ((1, 2), (0, 3)),
    }

    # The way we set this up.. the first tuple corresponds to
    # the Higgs candidate
    arr_pair_to_idx = np.array(
        [
            [0, 1, 2, 3],
            [0, 2, 1, 3],
            [0, 3, 1, 2],
            [2, 3, 0, 1],
            [1, 3, 0, 2],
            [1, 2, 0, 3],
        ]
    )

    drs = np.vstack(
        [
            hc_jets[:, ia[0]].deltaR(hc_jets[:, ia[1]]).to_numpy()
            for ia, ib in pair_to_idx.values()
        ]
    )
    evts["chosenPair"] = np.argmin(drs, axis=0)

    r = np.arange(len(hc_jets))
    idx = arr_pair_to_idx[evts.chosenPair]

    j11 = hc_jets[r, idx[:, 0]]
    j12 = hc_jets[r, idx[:, 1]]

    j21 = hc_jets[r, idx[:, 2]]
    j22 = hc_jets[r, idx[:, 3]]

    evts["dRjj_1"] = j11.deltaR(j12)
    evts["dRjj_2"] = j21.deltaR(j22)

    # Reconstruct the resonances
    Cand1 = j11 + j12
    Cand2 = j21 + j22

    return Cand1, Cand2


def defineCuts(evts, HC, SC, c1, c2, res1=0.1, res2=0.1):
    """
    Goal: Save some of the useful variables for defining the final SR:
    - dEta_SH
    - m_{S,H}
    - correct
    """

    SH = HC + SC
    evts["m_SH"] = SH.mass.to_numpy()
    evts["pt_SH"] = SH.pt.to_numpy()

    evts["pt_H"] = HC.pt.to_numpy()
    evts["eta_H"] = HC.eta.to_numpy()
    evts["phi_H"] = HC.phi.to_numpy()
    evts["m_H"] = HC.mass.to_numpy()

    evts["pt_S"] = SC.pt.to_numpy()
    evts["eta_S"] = SC.eta.to_numpy()
    evts["phi_S"] = SC.phi.to_numpy()
    evts["m_S"] = SC.mass.to_numpy()

    evts["dEta_SH"] = abs(HC.eta - SC.eta).to_numpy()
    evts["X_SH"] = getXhh(HC.mass, SC.mass, c1, c2, res1, res2).to_numpy()

    # Make a mask that's easier to check:
    if "correctPair" in evts.columns:
        evts["correct"] = evts["chosenPair"] == evts["correctPair"]


def pairAndProcess(
    inputFile: str,
    ntag=3,
    tagger="DL1d",
    WP=77,
    pairing="min_dH",
    cols: list = [],
    physicsSample=None,
    prodTag=None,
    year=None,
    nSelectedJets=4,
    pT_min=40,
    save=False,
    fileTag="",
    truth=False,
    outputDir="",
    bjetTrigSFs=False,
    truthJets=False,
):
    """
    Given a MNT, select the desired jets, do the GNN preprocessing, and apply
    the analysis cuts and return a df with a given # of btags in a specified
    kinematic region with only a subset of the columns.

    Inputs:
    - inputFile: The MNT file to evalutate on
    """

    """
    Step 1: Apply the jet selection
    """
    is_mc = False if ("data" in physicsSample) else True
    df, hc_jets = processDf(
        inputFile,
        min_btags=ntag,
        tagger=tagger,
        WP=WP,
        nJetsMax=nSelectedJets,
        pT_min=pT_min,
        year=year,
        mc=is_mc,
        truth=truth,
        truthJets=truthJets,
    )

    # outputFile = f"{outputDir}/files/df{fileTag}.parquet"

    # # Sometimes, we might not have any entries left
    # if df is None:
    #     print("No entries in file", inputFile)

    #     # Save an empty df
    #     table = pa.Table.from_pandas(pd.DataFrame())
    #     pq.write_table(table, outputFile)
    #     return None

    """
    Step 2: pair the HCs
    """
    assert nSelectedJets == 4

    if pairing == "min_dH":
        Cand1, Cand2 = do_min_dH(df, hc_jets)
    elif pairing == "min_dR":
        Cand1, Cand2 = do_min_dR(df, hc_jets)
    elif pairing == "min_dR_all":
        Cand1, Cand2 = do_min_dR_all(df, hc_jets)
    else:
        print("pairing", pairing, "not supported")
        raise NotImplementedError

    """
    Step 3: Apply the analysis cuts
    """
    # Infer the S mass from the string (if a signal)
    i = physicsSample.find("S")
    if i != -1:
        c2 = int(physicsSample[i + 1 :].split("_")[0])
        res2 = 0.1 * mH / c2
    else:
        c2 = 170  # it doesn't matter, j do an ex value
        res2 = 0.1

    if (pairing == "min_dR_all") and (c2 < mH):
        print(
            f"Running with min_dR_all and mS={c2} < mH={125}... swapping the S+H defns"
        )
        # The logic in min_dR_all assigs to HC to the smaller dRjj,
        # but the S has the smaller opening angle for mS < mH, so just swap
        # the two for this case!

        # Change the definition for chosenPair to match the defn for correctPair
        df["chosenPair"] = (df["chosenPair"] + 3) % 6

        defineCuts(df, Cand2, Cand1, mH, c2, res1=0.1, res2=res2)
    else:
        defineCuts(df, Cand1, Cand2, mH, c2, res1=0.1, res2=res2)

    for i in range(nSelectedJets):
        df[f"j{i}_pt"] = hc_jets[:, i].pt
        df[f"j{i}_eta"] = hc_jets[:, i].eta
        df[f"j{i}_phi"] = hc_jets[:, i].phi
        df[f"j{i}_E"] = hc_jets[:, i].energy

    """
    Step 4: Save and return the df
    """

    # if save:
    #     print("Save the parquet file", outputFile)
    #     table = pa.Table.from_pandas(df)
    #     pq.write_table(table, outputFile)

    return df
