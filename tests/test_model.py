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
    for f in ("development.csv", "learning_curves.csv", "metadata.json"):
        assert (out / f).exists()
    assert list((out / "brains").glob("*.npz"))

    from enrichsim.report import build_report
    path = build_report(out)
    text = path.read_text(encoding="utf-8")
    assert "<html" in text and "Méthode" in text
