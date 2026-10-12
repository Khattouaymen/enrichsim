"""Mesures « neuroanatomiques » du cerveau artificiel.

Correspondances avec les mesures histologiques des études d'enrichissement :

| Mesure du modèle               | Analogue biologique                                   |
|--------------------------------|-------------------------------------------------------|
| n_neurons                      | nombre de neurones (neurogenèse nette)                |
| n_synapses                     | densité synaptique / « épaisseur corticale »          |
| fan_in_mean                    | complexité dendritique (analyse de Sholl)             |
| synaptic_strength              | efficacité synaptique moyenne                         |
| coverage_*                     | territoire cortical dédié à chaque modalité           |
| dimensionality                 | richesse des représentations (ratio de participation) |
| sparseness                     | codage parcimonieux                                   |
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .brain import PlasticBrain
from .config import BrainConfig, EnvConfig
from .environment import HomeCage, N_ACTIONS


def modality_slices(cfg: BrainConfig) -> dict[str, np.ndarray]:
    """Indices des entrées appartenant à chaque modalité sensorielle."""
    side = 2 * cfg.view_radius + 1
    ch = 3 + cfg.object_features
    n_view = side * side * ch
    idx = np.arange(n_view).reshape(side * side, ch)
    return {
        "walls": idx[:, 0],
        "food": idx[:, 1],
        "social": idx[:, 2],
        "objects": idx[:, 3:].ravel(),
        "place": np.arange(n_view, n_view + cfg.place_cells ** 2),
    }


@lru_cache(maxsize=4)
def probe_set(cfg: BrainConfig, n: int = 400) -> np.ndarray:
    """Jeu d'observations de référence, identique pour tous les sujets.

    Il est généré dans un monde riche avec des objets JAMAIS vus pendant l'élevage, afin de
    mesurer la capacité de représentation sans favoriser une condition.
    """
    rng = np.random.default_rng(12345)
    probe_env = EnvConfig(name="probe", size=14, n_objects=12, object_pool=10_000,
                          novelty_period=50, n_conspecifics=3, inner_walls=5)
    world = HomeCage(probe_env, cfg, rng, type_offset=50_000)
    obs = []
    for _ in range(n * 3):
        world.step(int(rng.integers(N_ACTIONS)))
        obs.append(world.observe())
    X = np.array(obs[::3])
    X.setflags(write=False)
    return X


def participation_ratio(H: np.ndarray) -> float:
    """Dimensionnalité effective d'un ensemble d'activités (Gao & Ganguli, 2015)."""
    if H.shape[1] == 0:
        return 0.0
    C = np.cov(H, rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(np.atleast_2d(C)), 0, None)
    s = ev.sum()
    return float(s * s / (ev ** 2).sum()) if s > 0 else 0.0


@lru_cache(maxsize=4)
def object_probe(cfg: BrainConfig, n_objects: int = 12) -> np.ndarray:
    """Observations d'objets inédits présentés un à un, à la même place, dans une pièce vide."""
    from .environment import OBJECT, GridWorld
    rng = np.random.default_rng(0)
    X = []
    for k in range(n_objects):
        room = GridWorld(7, 7, cfg, rng)
        room.agent = np.array([3, 3])
        room.grid[3, 4] = OBJECT
        room.obj_type[3, 4] = 70_000 + k
        X.append(room.observe())
    X = np.array(X)
    X.setflags(write=False)
    return X


def pattern_separation(brain: PlasticBrain) -> float:
    """Séparation de patterns : distance euclidienne moyenne entre les représentations
    corticales d'objets différents vus au même endroit. Proche de 0 : le cortex
    « confond » les objets (cf. rôle du gyrus denté, Yassa & Stark, 2011)."""
    H = brain.activations(object_probe(brain.cfg))[:, brain.alive].astype(np.float64)
    d = np.sqrt(((H[:, None, :] - H[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(len(H), 1)
    return float(d[iu].mean())


def object_sensitivity(brain: PlasticBrain) -> float:
    """Amplitude de la réponse corticale propre aux objets (écart-type entre objets)."""
    H = brain.activations(object_probe(brain.cfg))[:, brain.alive]
    return float(H.std(0).mean())


def brain_metrics(brain: PlasticBrain) -> dict[str, float]:
    cfg = brain.cfg
    alive = brain.alive
    M = brain.M[alive]
    n_neurons = int(alive.sum())
    n_syn = int(M.sum())
    fan_in = M.sum(1) if n_neurons else np.zeros(1)
    out: dict[str, float] = {
        "n_neurons": n_neurons,
        "n_synapses": n_syn,
        "fan_in_mean": float(fan_in.mean()),
        "fan_in_sd": float(fan_in.std()),
        "synaptic_strength": float(np.abs(brain.W1[brain.M]).mean()) if n_syn else 0.0,
        "births": brain.births,
        "deaths": brain.deaths,
        "synapses_formed": brain.synapses_formed,
        "synapses_pruned": brain.synapses_pruned,
    }
    for name, idx in modality_slices(cfg).items():
        out[f"coverage_{name}"] = float(M[:, idx].sum() / max(1, n_neurons * len(idx)))

    X = probe_set(cfg)
    Hp = brain.activations(X)[:, alive]
    out["dimensionality"] = participation_ratio(Hp)
    active = Hp > 1e-6
    out["sparseness"] = float(1.0 - active.mean()) if Hp.size else 1.0
    out["responsive_neurons"] = float(active.any(0).mean()) if Hp.size else 0.0
    out["pattern_separation"] = pattern_separation(brain)
    out["object_sensitivity"] = object_sensitivity(brain)
    return out
