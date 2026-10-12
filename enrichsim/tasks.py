"""Batterie de tests comportementaux, inspirée des paradigmes utilisés chez le rongeur.

1. Champ ouvert (Hall, 1934)                 -> exploration, thigmotaxie (anxiété)
2. Reconnaissance d'objet nouveau (NOR)      -> mémoire de reconnaissance, attrait pour la nouveauté
   (Ennaceur & Delacour, 1988)
3. Labyrinthe de Hebb-Williams               -> apprentissage spatial / résolution de problèmes
   (Hebb & Williams, 1946 ; Rabinovitch & Rosvold, 1951)
4. Apprentissage inversé (reversal)          -> flexibilité cognitive, persévération

Tous les tests se déroulent dans des arènes que AUCUN groupe n'a vues pendant l'élevage.
"""
from __future__ import annotations

import numpy as np

from .config import TestConfig
from .environment import (MAZES, NOVEL_OBJECT_BASE, REVERSAL_GOAL, Maze, ObjectArena,
                          OpenField)
from .subject import Subject


def open_field(subject: Subject, cfg: TestConfig, size: int = 12) -> dict:
    arena = OpenField(size, subject.brain.cfg, subject.rng)
    center = {"n": 0}

    def on_step(world, info, out):
        y, x = world.agent
        if 3 <= y < size - 3 and 3 <= x < size - 3:
            center["n"] += 1

    st = subject.run(arena, cfg.open_field_steps, on_step=on_step)
    return {
        "of_exploration": len(st.visited) / (size * size),
        "of_center_time": center["n"] / st.steps,
    }


def novel_object_recognition(subject: Subject, cfg: TestConfig, size: int = 7) -> dict:
    """Plusieurs sessions indépendantes (paires d'objets différentes), moyennées."""
    sessions = [_nor_session(subject, cfg, size, k) for k in range(cfg.nor_sessions)]
    out = {}
    for key in sessions[0]:
        vals = [s[key] for s in sessions if not np.isnan(s[key])]
        out[key] = float(np.mean(vals)) if vals else np.nan
    return out


def _nor_session(subject: Subject, cfg: TestConfig, size: int, k: int) -> dict:
    fam = NOVEL_OBJECT_BASE + 2 * k + 1
    nov = NOVEL_OBJECT_BASE + 2 * k + 2
    novel_right = (subject.sid + k) % 2 == 0  # contrebalancement du côté de l'objet nouveau
    arena = ObjectArena(size, subject.brain.cfg, subject.rng, fam, fam)

    # familiarisation : deux objets identiques
    subject.run(arena, cfg.nor_familiarization)

    # test : un objet remplacé par un objet nouveau
    if novel_right:
        arena.set_objects(fam, nov)
    else:
        arena.set_objects(nov, fam)
    arena.agent = np.array([size - 1, size // 2])
    pos_nov = arena.right if novel_right else arena.left
    pos_fam = arena.left if novel_right else arena.right
    t = {"nov": 0, "fam": 0}
    pe = {"nov": [], "fam": []}

    def on_step(world, info, out):
        # exploration dirigée = contact actif avec l'objet (comme le « flairage » chez le rat)
        for key, pos in (("nov", pos_nov), ("fam", pos_fam)):
            if info.get("object_pos") == pos:
                t[key] += 1
                pe[key].append(out["object_novelty"])

    subject.run(arena, cfg.nor_test, on_step=on_step)
    total = t["nov"] + t["fam"]
    # indice non interprétable si l'objet n'est presque jamais exploré (critère usuel)
    di = (t["nov"] - t["fam"]) / total if total >= 5 else np.nan
    pe_n = float(np.mean(pe["nov"])) if pe["nov"] else np.nan
    pe_f = float(np.mean(pe["fam"])) if pe["fam"] else np.nan
    ok = pe["nov"] and pe["fam"] and (pe_n + pe_f) > 1e-9
    neural_di = (pe_n - pe_f) / (pe_n + pe_f) if ok else np.nan
    return {
        "nor_exploration": total / cfg.nor_test,
        "nor_di": di,
        "nor_neural_di": neural_di,
    }


def _maze_trials(subject: Subject, maze: Maze, n_trials: int, max_steps: int,
                 old_goal: tuple[int, int] | None = None) -> list[dict]:
    """Cotation inspirée de Hebb-Williams : une erreur = un déplacement qui éloigne du but
    (entrée dans une « zone d'erreur »). La latence (nombre de pas) est mesurée à part."""
    trials = []
    for trial in range(n_trials):
        maze.reset()
        count = {"errors": 0, "persev": 0}
        last = [tuple(int(v) for v in maze.agent)]

        def on_step(world, info, out):
            pos = (int(world.agent[0]), int(world.agent[1]))
            gd = maze.goal_distance
            if pos != last[0] and gd.get(pos, 99) >= gd.get(last[0], 99):
                count["errors"] += 1
            last[0] = pos
            if old_goal is not None and abs(pos[0] - old_goal[0]) + abs(pos[1] - old_goal[1]) <= 1:
                count["persev"] += 1

        st = subject.run(maze, max_steps, stop_on_food=True, on_step=on_step,
                         replay=max_steps, replay_passes=subject.brain.cfg.replay_passes)
        trials.append({
            "trial": trial + 1,
            "latency": st.steps,
            "errors": count["errors"],
            "success": st.reached,
            "perseveration": count["persev"],
        })
    return trials


def hebb_williams(subject: Subject, cfg: TestConfig) -> tuple[dict, list[dict]]:
    curves: list[dict] = []
    for prob in cfg.maze_problems:
        maze = Maze(prob, subject.brain.cfg, subject.rng)
        for tr in _maze_trials(subject, maze, cfg.maze_trials, cfg.maze_max_steps):
            curves.append({"task": "maze", "problem": prob, **tr})

    # apprentissage inversé sur le premier problème : le but change de place
    prob = cfg.maze_problems[0]
    maze = Maze(prob, subject.brain.cfg, subject.rng)
    old_goal = maze.goal
    rev = Maze(prob, subject.brain.cfg, subject.rng, goal=REVERSAL_GOAL[prob])
    for tr in _maze_trials(subject, rev, cfg.reversal_trials, cfg.maze_max_steps, old_goal=old_goal):
        curves.append({"task": "reversal", "problem": prob, **tr})

    maze_rows = [c for c in curves if c["task"] == "maze"]
    rev_rows = [c for c in curves if c["task"] == "reversal"]
    errors = np.array([[c["errors"] for c in maze_rows if c["problem"] == p] for p in cfg.maze_problems])
    latency = np.array([[c["latency"] for c in maze_rows if c["problem"] == p] for p in cfg.maze_problems])
    mean_curve = errors.mean(0)
    slope = float(np.polyfit(np.arange(len(mean_curve)), mean_curve, 1)[0])
    crit = []
    for p in cfg.maze_problems:
        rows = [c for c in maze_rows if c["problem"] == p]
        ok = [c["trial"] for c in rows if c["success"] and c["errors"] <= MAZE_CRITERION]
        crit.append(ok[0] if ok else cfg.maze_trials + 1)
    summary = {
        "maze_errors": float(errors.mean()),
        "maze_errors_first": float(errors[:, 0].mean()),
        "maze_errors_last": float(errors[:, -1].mean()),
        "maze_latency": float(latency.mean()),
        "maze_success": float(np.mean([c["success"] for c in maze_rows])),
        "maze_slope": slope,
        "maze_trials_to_criterion": float(np.mean(crit)),
        "reversal_errors": float(np.mean([c["errors"] for c in rev_rows])),
        "reversal_latency": float(np.mean([c["latency"] for c in rev_rows])),
        "reversal_perseveration": float(np.mean([c["perseveration"] for c in rev_rows])),
        "reversal_success": float(np.mean([c["success"] for c in rev_rows])),
    }
    return summary, curves


MAZE_CRITERION = 2  # critère d'apprentissage : essai réussi avec au plus 2 erreurs


def test_battery(subject: Subject, cfg: TestConfig) -> tuple[dict, list[dict]]:
    """Ordre fixe (du moins au plus stressant), comme dans les protocoles animaux."""
    res: dict = {}
    res.update(open_field(subject, cfg))
    res.update(novel_object_recognition(subject, cfg))
    maze_summary, curves = hebb_williams(subject, cfg)
    res.update(maze_summary)
    return res, curves


__all__ = ["test_battery", "open_field", "novel_object_recognition", "hebb_williams", "MAZES"]
