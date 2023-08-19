import h5py
import numpy as np
import awkward as ak

# dummy awkward array with 2 events and 2 jets and 3 jets in each
events = ak.Array(
    [
        {"jet_pt": [1, 2, 3, 4], "jet_eta": [1, 2, 3, 4]},
        {"jet_pt": [1, 2, 3], "jet_eta": [1, 2, 3]},
        {"jet_pt": [1, 2, 3, 4, 5, 6], "jet_eta": [1, 2, 3, 4, 5, 6]},
    ]
)
df = ak.to_dataframe(events)
df.to_hdf("test.h5", key="events", mode="w", format="table")
