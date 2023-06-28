"""
Module to collect some of the truth matching algs that I've been using
"""

import numpy as np
from itertools import combinations


def truthMatchJets(df, js, bs, njets=4):
    """
    Code to do the jet -> parton matching for defining the "correct pairing".

    Inputs:
    - df: df with the flattened inputs
    - js: awkward array (coffea-vector)
    - bs: awkward array for the b-quarks (altho there will be exactly 4 b-quarks)
    """

    hh = bs[:, 0] + bs[:, 1] + bs[:, 2] + bs[:, 3]
    df["truth_mhh_bs"] = hh.mass
    df["truth_pthh_bs"] = hh.pt

    # dRs is has shape (nEvt, # b-quarks, # jets)
    dRs = bs[:, :, np.newaxis].delta_r(js[:, np.newaxis, :])

    # Since there are 4 b-quarks, at this point the array is fixed dimensional
    idx = np.argmin(dRs, axis=-1).to_numpy().astype(int)
    dr_match = np.min(dRs, axis=-1).to_numpy()

    df[[f"b{i}_jidx" for i in range(4)]] = idx
    df[[f"b{i}_drMatch" for i in range(4)]] = dr_match

    # Save the bidx for each jet, where -1 will correspond to no truth matched jets
    for i in range(njets):
        df[f"j{i}_bidx"] = -1
        matched = np.sum(idx == i, axis=1).astype(bool)

        df.loc[matched, f"j{i}_bidx"] = np.argmax(idx[matched] == i, axis=1).astype(int)


def goodJets(df, Rmatch=0.3):
    """
    For a df, get a column for the # of unique jets with dRmatch < 0.3.
    """

    bi_jidx = df[[f"b{i}_jidx" for i in range(4)]].values
    bi_drMatch = df[[f"b{i}_drMatch" for i in range(4)]].values

    unique = np.ones_like(df.index).astype(bool)

    # Using np.unique didn't work - so I just ended up needing to check the
    # combinations manually
    for i, j in combinations(range(4), 2):
        mask = bi_jidx[:, i] == bi_jidx[:, j]
        unique[mask] = False

    dRmatch = ~np.sum(bi_drMatch > Rmatch, axis=1).astype(bool)

    df["unique"] = unique
    df["dRmatch"] = dRmatch
    df["goodJets"] = dRmatch & unique


def getCorrectPair(df, nJetsMax=6):
    df["correctPair"] = -1
    goodJets(df)

    N = len(df)

    for njets in range(4, nJetsMax + 1):
        # Get the mask for when we actually have njets
        if njets == nJetsMax:
            mask = df.njets >= njets
        else:
            mask = df.njets == njets
        mask = mask & (df.goodJets)

        i = 0

        for i0, i1, i2, i3 in combinations(range(njets), 4):
            """
            Check if the jets came from the same parent.

            Remember the jets are organized so that the first two bs are from the Higgs
            and the second two are from the S.
            """

            # Also - make sure all of the i0,i1,i2,i3 correspond to *matched* jets (i.e, not -1)
            matched = np.all(
                df[[f"j{ii}_bidx" for ii in [i0, i1, i2, i3]]].values >= 0, axis=1
            )

            # Pair 0: j0 and j1 make the Higgs, j2 and j3 make the S
            m0 = (
                ((df[f"j{i0}_bidx"] == 0) & (df[f"j{i1}_bidx"] == 1))
                | ((df[f"j{i0}_bidx"] == 1) & (df[f"j{i1}_bidx"] == 0))
            ) & (
                ((df[f"j{i2}_bidx"] == 2) & (df[f"j{i3}_bidx"] == 3))
                | ((df[f"j{i2}_bidx"] == 3) & (df[f"j{i3}_bidx"] == 2))
            ).values

            # Pair 1: j0 and j2 make the Higgs, j1 and j3 make the S
            m1 = (
                ((df[f"j{i0}_bidx"] == 0) & (df[f"j{i2}_bidx"] == 1))
                | ((df[f"j{i0}_bidx"] == 1) & (df[f"j{i2}_bidx"] == 0))
            ) & (
                ((df[f"j{i1}_bidx"] == 2) & (df[f"j{i3}_bidx"] == 3))
                | ((df[f"j{i1}_bidx"] == 3) & (df[f"j{i3}_bidx"] == 2))
            ).values

            # Pair 2: j0 and j3 make the Higgs, j1 and j2 make the S
            m2 = (
                ((df[f"j{i0}_bidx"] == 0) & (df[f"j{i3}_bidx"] == 1))
                | ((df[f"j{i0}_bidx"] == 1) & (df[f"j{i3}_bidx"] == 0))
            ) & (
                ((df[f"j{i1}_bidx"] == 2) & (df[f"j{i2}_bidx"] == 3))
                | ((df[f"j{i1}_bidx"] == 3) & (df[f"j{i2}_bidx"] == 2))
            ).values

            # Pair 3: j2 and j3 make the Higgs, j0 and j1 make the S
            m3 = (
                ((df[f"j{i2}_bidx"] == 0) & (df[f"j{i3}_bidx"] == 1))
                | ((df[f"j{i2}_bidx"] == 1) & (df[f"j{i3}_bidx"] == 0))
            ) & (
                ((df[f"j{i0}_bidx"] == 2) & (df[f"j{i1}_bidx"] == 3))
                | ((df[f"j{i0}_bidx"] == 3) & (df[f"j{i1}_bidx"] == 2))
            ).values

            # Pair 4: j1 and j3 make the Higgs, j0 and j2 make the S
            m4 = (
                ((df[f"j{i1}_bidx"] == 0) & (df[f"j{i3}_bidx"] == 1))
                | ((df[f"j{i1}_bidx"] == 1) & (df[f"j{i3}_bidx"] == 0))
            ) & (
                ((df[f"j{i0}_bidx"] == 2) & (df[f"j{i2}_bidx"] == 3))
                | ((df[f"j{i0}_bidx"] == 3) & (df[f"j{i2}_bidx"] == 2))
            ).values

            # Pair 5: j1 and j2 make the Higgs, j0 and j3 make the S
            m5 = (
                ((df[f"j{i1}_bidx"] == 0) & (df[f"j{i2}_bidx"] == 1))
                | ((df[f"j{i1}_bidx"] == 1) & (df[f"j{i2}_bidx"] == 0))
            ) & (
                ((df[f"j{i0}_bidx"] == 2) & (df[f"j{i3}_bidx"] == 3))
                | ((df[f"j{i0}_bidx"] == 3) & (df[f"j{i3}_bidx"] == 2))
            ).values

            # Loop through the pairings
            df.loc[mask & matched & m0, "correctPair"] = i
            df.loc[mask & matched & m1, "correctPair"] = i + 1
            df.loc[mask & matched & m2, "correctPair"] = i + 2

            df.loc[mask & matched & m3, "correctPair"] = i + 3
            df.loc[mask & matched & m4, "correctPair"] = i + 4
            df.loc[mask & matched & m5, "correctPair"] = i + 5

            i += 3


def getTruthJets(tree, df, nJetsMax, entrystart=None, entrystop=None):
    """
    Goal: Save the truth jets matching to reco (analysis) jets
    """

    # Step 1: Load in the truth jet info
    tcols = [f"truthjet_antikt4_{v}" for v in ["pt", "eta", "phi", "m"]]
    # Warning: The truthJets have pt and m still in MeV
    truthJets = tree.arrays(tcols, entrystart=entrystart, entrystop=entrystop)

    tvec = TLorentzVectorArray.from_ptetaphim(
        0.001 * truthJets[b"truthjet_antikt4_pt"],
        truthJets[b"truthjet_antikt4_eta"],
        truthJets[b"truthjet_antikt4_phi"],
        0.001 * truthJets[b"truthjet_antikt4_m"],
    )

    # Step 2: Loop over the reco jets and save the matched truth jet
    for i in range(nJetsMax):
        ji = TLorentzVectorArray.from_ptetaphie(
            *df[[f"j{i}_{v}" for v in ["pt", "eta", "phi", "E"]]].values.T
        )

        # Get the dR matrix
        out = tvec.delta_r(ji)

        idx = out.argmin().flatten()

        df[f"j{i}_tidx"] = out.argmin().flatten()
        df[f"j{i}_tdR"] = out.min()

        truthJet = tvec[range(len(df.index)), idx]

        # Save the rest of the 4-vector info
        df[f"j{i}_tpt"] = truthJet.pt
        df[f"j{i}_teta"] = truthJet.eta
        df[f"j{i}_tphi"] = truthJet.phi
        df[f"j{i}_tE"] = truthJet.E

    # Step 3: Save columns for if we matched to unique jets + (later)
