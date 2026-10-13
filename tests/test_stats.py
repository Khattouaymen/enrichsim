"""Tests des fonctions statistiques sur des données dont on connaît la réponse."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from enrichsim import stats as S


def make_groups(means, n=30, sd=1.0, seed=0, var="brain_n_synapses"):
    rng = np.random.default_rng(seed)
    rows = []
    for g, m in means.items():
        for i in range(n):
            rows.append({"group": g, "subject": i, var: rng.normal(m, sd)})
    return pd.DataFrame(rows)


def test_fmt_p_apa():
    assert S.fmt_p(0.0004) == "p < .001"
    assert S.fmt_p(0.0153) == "p = .015"
    assert S.fmt_p(0.5) == "p = .500"


def test_holm_matches_definition():
    p = [0.01, 0.04, 0.03, np.nan]
    out = S.holm(p)
    assert out[0] == pytest.approx(0.03)
    assert out[2] == pytest.approx(0.06)
    assert out[1] == pytest.approx(0.06)
    assert np.isnan(out[3])


def test_hedges_g_sign_and_size():
    a = np.array([2, 3, 4, 5, 6.0])
    b = a - 1.58113883  # écart = 1 ET poolé
    g = S.hedges_g(a, b)
    assert 0.85 < g < 1.0


def test_large_effect_detected():
    df = make_groups({"IC": 0, "SC": 0.5, "EC": 2})
    r = S.compare_groups(df, "brain_n_synapses")
    assert r.p < 0.001 and r.eta2 > 0.4
    ec_ic = [p for p in r.posthoc if {p["A"], p["B"]} == {"IC", "EC"}][0]
    assert ec_ic["p"] < 0.001


def test_null_effect_not_detected():
    df = make_groups({"IC": 0, "SC": 0, "EC": 0}, seed=3)
    r = S.compare_groups(df, "brain_n_synapses")
    assert r.p > 0.05


def test_anova_matches_scipy():
    df = make_groups({"A": 0, "B": 0.3, "C": 0.6}, seed=5)
    r = S.compare_groups(df, "brain_n_synapses")
    samples = [df.loc[df.group == g, "brain_n_synapses"] for g in "ABC"]
    if r.test == "ANOVA":
        assert r.p == pytest.approx(sps.f_oneway(*samples).pvalue)


def test_welch_chosen_when_variances_differ():
    rng = np.random.default_rng(1)
    rows = [{"group": g, "subject": i, "x": rng.normal(0, sd)}
            for g, sd in (("A", 0.2), ("B", 1.0), ("C", 5.0)) for i in range(40)]
    r = S.compare_groups(pd.DataFrame(rows), "x")
    assert r.test == "Welch" and r.posthoc_method == "Games-Howell"


def test_kruskal_chosen_when_not_normal():
    rng = np.random.default_rng(2)
    rows = [{"group": g, "subject": i, "x": rng.exponential(1.0) ** 3}
            for g in "ABC" for i in range(40)]
    r = S.compare_groups(pd.DataFrame(rows), "x")
    assert r.test == "Kruskal-Wallis"


def test_constant_variable_skipped():
    df = pd.DataFrame({"group": list("AABB"), "subject": [0, 1, 0, 1], "x": [1.0] * 4})
    assert S.compare_groups(df, "x") is None


def test_dose_regression_slope():
    rng = np.random.default_rng(0)
    lv = np.repeat([0, .25, .5, .75, 1], 20)
    df = pd.DataFrame({"level": lv, "brain_n_synapses": 3 * lv + rng.normal(0, .5, len(lv))})
    out = S.analyze_dose(df, ["brain_n_synapses"])
    assert out.slope.iloc[0] == pytest.approx(3, abs=0.4)
    assert out.p.iloc[0] < 0.001


def test_timing_main_effect():
    rng = np.random.default_rng(0)
    rows = [{"early": e, "late": la, "group": f"{e}{la}", "subject": i,
             "brain_n_synapses": 2 * e + rng.normal()} for e in (0, 1) for la in (0, 1) for i in range(20)]
    out = S.analyze_timing(pd.DataFrame(rows), ["brain_n_synapses"])
    assert out.p_early.iloc[0] < 0.001
    assert out.p_late.iloc[0] > 0.01


def test_mixed_model_detects_slope_difference():
    rng = np.random.default_rng(0)
    rows = []
    for g, slope in (("IC", 0.0), ("EC", -2.0)):
        for s in range(15):
            u = rng.normal(0, 1)
            for t in range(1, 9):
                rows.append({"task": "maze", "problem": 0, "group": g, "subject": s, "trial": t,
                             "errors": 20 + u + slope * t + rng.normal(0, 1)})
    m = S.learning_mixed_model(pd.DataFrame(rows), order=["IC", "EC"])
    assert m["wald_p"] < 0.001


def test_compare_protocols_detects_interaction():
    """Le protocole B améliore le score du seul groupe EC : interaction groupe x protocole."""
    rng = np.random.default_rng(0)
    rows_a, rows_b = [], []
    for g in ("IC", "SC", "EC"):
        for i in range(25):
            base = rng.normal(0, 1)
            common = {"group": g, "subject": i, "brain_n_synapses": 100 + i}
            rows_a.append({**common, "maze_errors": base})
            rows_b.append({**common, "maze_errors": base + (3.0 if g == "EC" else 0.0) + rng.normal(0, 0.3)})
    out, same = S.compare_protocols(pd.DataFrame(rows_a), pd.DataFrame(rows_b), ["maze_errors"],
                                    ["IC", "SC", "EC"])
    assert same is True
    r = out.iloc[0]
    assert r.inter_p < 0.001
    assert r.mean_diff == pytest.approx(1.0, abs=0.2)


def test_compare_protocols_flags_different_rearing():
    a = pd.DataFrame({"group": ["IC", "EC"] * 3, "subject": [0, 0, 1, 1, 2, 2],
                      "brain_n_synapses": [1, 2, 3, 4, 5, 6], "maze_errors": [1.0, 2, 3, 4, 5, 6]})
    b = a.assign(brain_n_synapses=a.brain_n_synapses + 1)
    _, same = S.compare_protocols(a, b, ["maze_errors"], ["IC", "EC"])
    assert same is False
