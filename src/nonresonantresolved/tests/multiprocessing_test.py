from multiprocessing import Lock, Manager, Pool
from src.nonresonantresolved.hist import Histogram


def f(L, i):
    hists0 = L[0]
    hists1 = L[1]
    with Lock():
        # fill hists, mimicking what is happening in hh4b_non_res_res_make_hists.py when processing a batch
        for i in range(100):
            hists0[0].fill([1, 2, 3, 4])
            hists1[0].fill([1, 2, 3, 4])
        # NOTE: important: copy the hists back (otherwise parent process won't see the changes)
        L[0] = hists0
        L[1] = hists1


def test_multiprocessing():
    manager = Manager()
    dict = {
        0: [Histogram("first_hist", [0, 5], 5)],
        1: [Histogram("second_hist", [0, 5], 5)],
    }
    dict = manager.dict(dict)

    with Pool(2) as p:
        p.starmap(f, [(dict, 1), (dict, 2)])

    for key, value in dict.items():
        assert all(val == 100.0 for val in value[0].values)


def run_tests():
    test_multiprocessing()


if __name__ == "__main__":
    run_tests()
