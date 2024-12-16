import vector
import numpy as np
import awkward as ak
import itertools as it
import onnxruntime as ort

vector.register_awkward()


def make_4jet_comb_array(a, op):
    fourpairs = list(it.combinations(range(4), 2))
    return np.transpose(ak.Array(op(a[:, i], a[:, j]) for i, j in fourpairs))


def pad(x: ak.Array, pad_end: int = 20, pad_with: float = np.nan):
    x = ak.pad_none(x, pad_end, axis=1, clip=True)
    x = ak.fill_none(x, pad_with)
    return ak.from_regular(x)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    exp_x = np.exp(
        x - np.max(x, axis=1, keepdims=True)
    )  # Subtract max for numerical stability
    return exp_x / np.sum(
        exp_x, axis=1, keepdims=True
    )  # Use keepdims=True to maintain the shape


def get_deepset_inputs(dataset, max_jets):
    jet_feature_names = [
        "jet_px",
        "jet_py",
        "jet_pz",
        "jet_btag",
        "bb_RmH",
        "bb_dR",
        "bb_dEta",
    ]
    event_feature_names = ["m_4b"]

    jet_p4 = ak.zip(
        {k: dataset[f"jet_{k}"] for k in ["pt", "eta", "phi", "mass"]},
        with_name="Momentum4D",
    )
    four_bjets_p4 = jet_p4[dataset.hh_jet_idx]

    # jet-level features
    X_JET = dataset[["jet_btag"]]
    X_JET["jet_px"] = jet_p4.px
    X_JET["jet_py"] = jet_p4.py
    X_JET["jet_pz"] = jet_p4.pz
    X_JET["bb_RmH"] = (
        make_4jet_comb_array(four_bjets_p4, lambda x, y: (x + y).mass) / 125.0
    )
    X_JET["bb_dR"] = make_4jet_comb_array(four_bjets_p4, lambda x, y: x.deltaR(y))
    X_JET["bb_dEta"] = make_4jet_comb_array(
        four_bjets_p4, lambda x, y: abs(x.eta - y.eta)
    )
    for feature in X_JET.fields:
        X_JET[feature] = pad(X_JET[feature], pad_end=max_jets, pad_with=-9999)
    X_JET = np.transpose(
        [X_JET[f].to_numpy().astype(np.float32) for f in X_JET.fields], axes=(1, 0, 2)
    )

    # Event-level features
    X_EVENT = ak.zip(
        {
            "m_4b": (
                four_bjets_p4[:, 0]
                + four_bjets_p4[:, 1]
                + four_bjets_p4[:, 2]
                + four_bjets_p4[:, 3]
            ).mass
        }
    )
    X_EVENT = np.transpose(
        [X_EVENT[f].to_numpy().astype(np.float32) for f in X_EVENT.fields]
    )

    return X_JET, X_EVENT


# Load the ONNX model
def get_inferences(
    ort_session: ort.InferenceSession, input_jet: np.ndarray, input_event: np.ndarray
):
    # compute ONNX Runtime output prediction
    input_jet_name = ort_session.get_inputs()[0].name
    input_event_name = ort_session.get_inputs()[1].name
    ort_inputs = {
        input_jet_name: input_jet,
        input_event_name: input_event,
    }
    ort_outs = ort_session.run(None, ort_inputs)
    probabilities = softmax(ort_outs[0])
    discriminant = get_score(
        y_pred=probabilities,
        class_labels=["HH", "QCD", "ttbar"],
        main_class="HH",
        frac_dict={"QCD": 0.92, "ttbar": 0.08},
    )
    return discriminant


def get_score(
    y_pred: np.ndarray,
    class_labels: list,
    main_class: str,
    frac_dict: dict,
) -> np.ndarray:
    """
    Similar to CalcDiscValues but uses directly the output of the
    NN (shape: (n_jets, nClasses)) for calculation.

    Parameters
    ----------
    y_pred : numpy.ndarray
        The prediction output of the NN.
    class_labels : list
        A list of the class_labels which are used.
    main_class : str
        The main discriminant class. For HH-tagging obviously "HH".
    frac_dict : dict
        A dict with the respective fractions for each class provided
        except main_class.
    use_keras_backend : bool
        Decide, if the values are calculated with the keras backend
        or numpy (Keras is needed for the saliency maps).

    Returns
    -------
    disc_score : numpy.ndarray
        Discriminant Score for the jets provided.

    Raises
    ------
    KeyError
        If for the given class label no frac_dict entry is given

    Examples
    --------
    >>> y_pred = np.array(
    ...     [
    ...         [0.1, 0.1, 0.8],
    ...         [0.0, 0.1, 0.9],
    ...         [0.2, 0.6, 0.2],
    ...         [0.1, 0.8, 0.1],
    ...     ]
    ... )
    array([[0.1, 0.1, 0.8],
           [0. , 0.1, 0.9],
           [0.2, 0.6, 0.2],
           [0.1, 0.8, 0.1]])

    >>> class_labels = ["HH", "QCD", "ttbar"]
    ['HH', 'QCD', 'ttbar']

    >>> main_class = "HH"
    'HH'

    >>> frac_dict = {"ttbar": 0.018, "QCD": 0.982}
    {'ttbar': 0.018, 'QCD': 0.982}

    Now we can call the function which will return the discriminant values
    for the given jets based on their given NN outputs (y_pred).

    >>> disc_scores = get_score(
    ...     y_pred=y_pred,
    ...     class_labels=class_labels,
    ...     main_class=main_class,
    ...     frac_dict=frac_dict,
    ... )
    [2.07944154, 6.21460804, -0.03536714, -0.11867153]
    """

    # Check if y_pred and class_labels have the same shapes
    assert np.shape(y_pred)[1] == len(class_labels)

    # Assert that frac_dict has class_labels as keys except main_class
    assert set(frac_dict.keys()) == set(class_labels) - set([main_class])

    # Ensure that y_pred is full precision (float32)
    if isinstance(y_pred, np.ndarray):
        y_pred = y_pred.astype("float32")

    # Get class_labels_idx for class_labels
    class_labels_idx = {label: idx for idx, label in enumerate(class_labels)}

    # Get list of class_labels without main_class
    class_labels_wo_main = [
        class_label for class_label in class_labels if class_label != main_class
    ]

    # Init denominator of disc_score and add_small
    denominator = 0
    numerator = 0
    add_small = 1e-10

    # Calculate numerator of disc_score
    numerator += y_pred[:, class_labels_idx[main_class]]
    numerator += add_small

    # Calculate denominator of disc_score
    for class_label in class_labels_wo_main:
        denominator += frac_dict[class_label] * y_pred[:, class_labels_idx[class_label]]

    denominator += add_small

    # Calculate final disc_score and return it
    disc_value = np.log(numerator / denominator)

    return disc_value
