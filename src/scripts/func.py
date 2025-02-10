

from src.solver import TACOORIG, TACONL, TACOL


def test_orig(qig, mems, comms, W):
    model = TACOORIG(qig, mems, comms, W)
    model.build()
    model.solve()
    return model.get_results()