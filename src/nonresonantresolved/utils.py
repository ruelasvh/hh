import re
from functools import reduce


def find_hist(
    iteratable,
    pred=None,
    default=False,
):
    """Returns the first true value in the iterable.

    If no true value is found, returns *default*

    If *pred* is not None, returns the first item
    for which pred(item) is true.

    """
    return next(filter(pred, iteratable), default)


def find_all_hists(hists, delimeter):
    """Returns the list of hists that match the delimeter"""
    prog = re.compile(delimeter)
    return list(filter(lambda h: prog.match(h.name), hists))


def get_all_trigs_or(events, trigs, skip_trig=None):
    """Returns the OR decision of all trigs except skip_trig"""
    trigs = list(filter(lambda trig: trig != skip_trig, trigs))
    return reduce(
        lambda acc, it: acc | events[it],
        trigs,
        events[trigs[0]],
    )


nth = {1: "first", 2: "second", 3: "third", 4: "fourth"}

kin_vars = {"pt": "$p_T$", "eta": "$\eta$", "phi": "$\phi$", "m": "$m$"}

inv_GeV = 1 / 1_000
