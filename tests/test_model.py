"""Tests de validité du modèle (environnements, cerveau, métriques, déterminisme)."""
import numpy as np
import pandas as pd
import pytest

from enrichsim.config import CONDITIONS, EC, IC, BrainConfig, Phase, ablate, interpolate_enrichment
from enrichsim.config import TestConfig as BatteryConfig
from enrichsim.environment import MAZES, REVERSAL_GOAL, HomeCage, Maze, obs_dim
from enrichsim.experiment import run_experiment
from enrichsim.metrics import brain_metrics
from enrichsim.subject import Subject
from enrichsim.tasks import TEST_STAGES
from enrichsim.tasks import test_battery as run_battery

CFG = BrainConfig()
FAST_TESTS = BatteryConfig(open_field_steps=50, nor_sessions=1, nor_familiarization=30, nor_test=40,
                        maze_trials=2, maze_max_steps=40, maze_problems=(0,), reversal_trials=1)


@pytest.mark.parametrize("problem", range(len(MAZES)))
def test_mazes_are_solvable(problem):
    rng = np.random.default_rng(0)
    m = Maze(problem, CFG, rng)
    assert m.optimal > 0, "le but doit être accessible depuis le départ"
    rev = Maze(problem, CFG, rng, goal=REVERSAL_GOAL[problem])
    assert rev.optimal > 0, "le but de l'inversion doit être accessible"
    assert rev.goal != m.goal


@pytest.mark.parametrize("env", list(CONDITIONS.values()) + [ablate(EC, c) for c in
                                                             ("objects", "novelty", "social", "space")])
def test_home_cages_connected_and_observable(env):
    cage = HomeCage(env, CFG, np.random.default_rng(1))
    assert cage.connected()
    x = cage.observe()
    assert x.shape == (obs_dim(CFG),)
    assert np.all(np.isfinite(x))


def test_enrichment_interpolation_bounds():
    assert interpolate_enrichment(0.0).size == IC.size
    assert interpolate_enrichment(1.0).size == EC.size
    assert interpolate_enrichment(1.0).n_objects == EC.n_objects
    sizes = [interpolate_enrichment(lv).size for lv in np.linspace(0, 1, 5)]
    assert sizes == sorted(sizes)


def test_unknown_ablation_rejected():
    with pytest.raises(ValueError):
        ablate(EC, "inconnu")


def _develop(seed, env=EC, steps=600):
    s = Subject(0, CFG, seed)
    traj = s.develop([Phase(env, steps)], log_every=300)
    return s, traj


def test_determinism_same_seed():
    a, ta = _develop(42)
    b, tb = _develop(42)
    np.testing.assert_array_equal(a.brain.W1, b.brain.W1)
    np.testing.assert_array_equal(a.brain.M, b.brain.M)
    assert ta[-1]["n_synapses"] == tb[-1]["n_synapses"]


def test_different_seeds_differ():
    a, _ = _develop(1)
    b, _ = _develop(2)
    assert not np.array_equal(a.brain.W1, b.brain.W1)


def test_matched_design_same_initial_brain():
    """Plan apparié : même graine -> même cerveau à la naissance, quelle que soit la cage."""
    a = Subject(0, CFG, 7)
    b = Subject(0, CFG, 7)
    np.testing.assert_array_equal(a.brain.W1, b.brain.W1)


@pytest.mark.parametrize("env", [IC, EC])
def test_metrics_valid_no_nan(env):
    s, traj = _develop(3, env)
    m = brain_metrics(s.brain)
    for k, v in m.items():
        assert np.isfinite(v), k
    assert 0 < m["n_neurons"] <= CFG.n_max
    assert m["n_synapses"] <= m["n_neurons"] * s.brain.M.shape[1]
    for k in ("coverage_walls", "coverage_food", "coverage_social", "coverage_objects", "coverage_place",
              "sparseness", "responsive_neurons"):
        assert 0.0 <= m[k] <= 1.0, k
    assert np.all(np.isfinite(s.brain.W1))
    assert not np.any(s.brain.W1[~s.brain.M]), "aucun poids hors des synapses existantes"


def test_full_pipeline_small(tmp_path):
    """Mini-expérience de bout en bout : fichiers produits, pas de NaN anatomique, rapport."""
    out = run_experiment("main", n_subjects=2, steps=300, out_dir=tmp_path / "main", workers=1,
                         seed=1, tests=FAST_TESTS, log_every=150)
    df = pd.read_csv(out / "subjects.csv")
    assert len(df) == 6 and set(df.group) == {"IC", "SC", "EC"}
    anat = [c for c in df.columns if c.startswith("brain_") and c not in
            ("brain_food_rate", "brain_touch_rate", "brain_distinct_objects", "brain_exploration",
             "brain_prediction_error")]
    assert not df[anat].isna().any().any()
    for f in ("development.csv", "learning_curves.csv", "test_development.csv", "metadata.json"):
        assert (out / f).exists()
    # cerveaux enregistrés en fin d'élevage ET après les tests
    for g in ("IC", "SC", "EC"):
        end = np.load(out / "brains" / f"{g}_s0.npz")
        post = np.load(out / "brains" / f"{g}_s0_apres_tests.npz")
        assert str(end["stage"]) == "fin d'élevage" and str(post["stage"]) == "après les tests"
        row = df[(df.group == g) & (df.subject == 0)].iloc[0]
        assert int(end["M"][end["alive"]].sum()) == row.brain_n_synapses

    from enrichsim.report import build_report
    path = build_report(out)
    text = path.read_text(encoding="utf-8")
    assert "<html" in text and "Méthode" in text


def test_state_is_a_copy():
    s, _ = _develop(5, steps=300)
    snap = s.brain.state()
    s.brain.W1[:] = 0.0
    assert np.any(snap["W1"] != 0.0)


LONG_TESTS = BatteryConfig(open_field_steps=600, nor_sessions=1, nor_familiarization=30, nor_test=40,
                           maze_trials=2, maze_max_steps=40, maze_problems=(0,), reversal_trials=1)


@pytest.mark.parametrize("freeze", [True, False])
def test_freeze_structure_during_tests(freeze):
    """Architecture figée pendant les tests si demandé ; elle change sinon (plasticité active)."""
    s, _ = _develop(11, EC, steps=600)
    M0, alive0 = s.brain.M.copy(), s.brain.alive.copy()
    W0 = s.brain.W1.copy()
    cfg = BatteryConfig(**{**LONG_TESTS.__dict__, "freeze_structure": freeze})
    _, _, stages = run_battery(s, cfg)
    assert [st["stage_label"] for st in stages] == TEST_STAGES
    if freeze:
        np.testing.assert_array_equal(s.brain.M, M0)
        np.testing.assert_array_equal(s.brain.alive, alive0)
        assert len({st["n_synapses"] for st in stages}) == 1
        assert not np.array_equal(s.brain.W1, W0), "les poids continuent d'apprendre"
    else:
        assert not np.array_equal(s.brain.M, M0)
    assert s.brain.structure_frozen is False, "le gel ne doit pas survivre à la batterie de tests"


def test_stage_zero_is_end_of_rearing():
    s, traj = _develop(13, IC, steps=600)
    _, _, stages = run_battery(s, FAST_TESTS)
    assert stages[0]["n_synapses"] == traj[-1]["n_synapses"]
    assert stages[0]["pattern_separation"] == pytest.approx(traj[-1]["pattern_separation"])


def test_control_design_same_rearing(tmp_path):
    """Le plan de contrôle reprend exactement l'élevage du plan principal (mêmes graines)."""
    kw = dict(n_subjects=2, steps=300, workers=1, seed=3, tests=FAST_TESTS, log_every=150)
    a = pd.read_csv(run_experiment("main", out_dir=tmp_path / "main", **kw) / "subjects.csv")
    out = run_experiment("control", out_dir=tmp_path / "control", **kw)
    b = pd.read_csv(out / "subjects.csv")
    anat = [c for c in a.columns if c.startswith("brain_")]
    pd.testing.assert_frame_equal(a[anat], b[anat])
    import json
    assert json.loads((out / "metadata.json").read_text(encoding="utf-8"))["tests"]["freeze_structure"]
    st = pd.read_csv(out / "test_development.csv")
    assert (st.groupby(["group", "subject"])["n_synapses"].nunique() == 1).all()

    from enrichsim.stats import analyze
    r = analyze(out, save=False)
    assert r["same_rearing"] is True
    assert not r["vs_main"].empty
