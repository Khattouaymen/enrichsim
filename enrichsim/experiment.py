"""Protocoles expérimentaux et exécution parallèle.

Quatre expériences sont définies :

1. `main`     — IC vs SC vs EC (réplique de Rosenzweig et al., 1962-1972).
2. `dose`     — relation dose-réponse : 5 niveaux d'enrichissement de 0 à 1.
3. `timing`   — période critique et réversibilité : plan factoriel 2 x 2
                (enrichissement précoce oui/non) x (enrichissement tardif oui/non),
                avec une plasticité qui décroît avec l'âge.
4. `ablation` — quelle composante de l'enrichissement compte ? (objets, nouveauté,
                social, espace), chacune retirée séparément de EC.

Plan apparié : le sujet n°i de chaque groupe démarre avec EXACTEMENT le même cerveau
initial (même graine), comme des individus d'une même portée répartis entre les
conditions (« split-litter design »). Les différences finales ne peuvent donc venir que
de l'environnement.
"""
from __future__ import annotations

import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import __version__
from .config import (CONDITIONS, EC, IC, BrainConfig, Group, Phase, TestConfig, ablate,
                     interpolate_enrichment)
from .subject import Subject
from .tasks import test_battery

# --------------------------------------------------------------------------- plans expérimentaux


def design_main(steps: int, brain: BrainConfig) -> list[Group]:
    return [Group(name, [Phase(env, steps)], brain, {"condition": name})
            for name, env in CONDITIONS.items()]


def design_dose(steps: int, brain: BrainConfig) -> list[Group]:
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    return [Group(f"E{lv:.2f}", [Phase(interpolate_enrichment(lv), steps)], brain, {"level": lv})
            for lv in levels]


def design_timing(steps: int, brain: BrainConfig) -> list[Group]:
    brain = replace(brain, critical_period=True, plasticity_tau=steps / 1.5)
    half = steps // 2
    groups = []
    for early in (False, True):
        for late in (False, True):
            e1 = EC if early else IC
            e2 = EC if late else IC
            groups.append(Group(f"{e1.name}→{e2.name}", [Phase(e1, half), Phase(e2, steps - half)],
                                brain, {"early": int(early), "late": int(late)}))
    return groups


def design_ablation(steps: int, brain: BrainConfig) -> list[Group]:
    groups = [Group("EC", [Phase(EC, steps)], brain, {"removed": "aucune"})]
    for comp, label in (("objects", "objets"), ("novelty", "nouveauté"),
                        ("social", "social"), ("space", "espace")):
        env = ablate(EC, comp)
        groups.append(Group(f"EC sans {label}", [Phase(env, steps)], brain, {"removed": label}))
    groups.append(Group("IC", [Phase(IC, steps)], brain, {"removed": "tout"}))
    return groups


DESIGNS: dict[str, Callable[[int, BrainConfig], list[Group]]] = {
    "main": design_main,
    "dose": design_dose,
    "timing": design_timing,
    "ablation": design_ablation,
}

DESCRIPTIONS = {
    "main": "Effet de l'environnement d'élevage (IC / SC / EC)",
    "dose": "Relation dose-réponse de l'enrichissement",
    "timing": "Période critique : enrichissement précoce vs tardif (2 x 2)",
    "ablation": "Contribution de chaque composante de l'enrichissement",
}


# --------------------------------------------------------------------------- exécution d'un sujet

def run_subject(group: Group, index: int, seed: int, tests: TestConfig,
                log_every: int, keep_brain: bool) -> dict:
    t0 = time.time()
    subject = Subject(index, group.brain, seed)
    trajectory = subject.develop(group.phases, log_every=log_every)
    anatomy = {f"brain_{k}": v for k, v in trajectory[-1].items()
               if k not in ("age", "phase", "env")}
    behaviour, curves = test_battery(subject, tests)
    row = {"group": group.name, "subject": index, "seed": seed, **group.factors,
           **anatomy, **behaviour, "runtime_s": round(time.time() - t0, 2)}
    for r in trajectory:
        r.update(group=group.name, subject=index)
    for c in curves:
        c.update(group=group.name, subject=index)
    out = {"row": row, "trajectory": trajectory, "curves": curves}
    if keep_brain:
        out["brain"] = {k: np.asarray(v) for k, v in subject.brain.state().items()}
    return out


def _job(args):
    return run_subject(*args)


def run_experiment(design: str, n_subjects: int = 20, steps: int = 20_000,
                   out_dir: str | Path = "results", workers: int | None = None,
                   seed: int = 2026, tests: TestConfig | None = None,
                   brain: BrainConfig | None = None, log_every: int | None = None,
                   progress: Callable[[int, int, str], None] | None = None) -> Path:
    """Lance une expérience complète et enregistre les données brutes.

    Retourne le dossier de résultats contenant :
      subjects.csv         une ligne par sujet (anatomie + comportement)
      development.csv      trajectoires développementales
      learning_curves.csv  performances essai par essai (labyrinthe, inversion)
      brains/*.npz         cerveaux de quelques sujets (visualisation)
      metadata.json        paramètres complets (reproductibilité)
    """
    tests = tests or TestConfig()
    brain = brain or BrainConfig()
    log_every = log_every or max(500, steps // 20)
    groups = DESIGNS[design](steps, brain)
    out = Path(out_dir)
    (out / "brains").mkdir(parents=True, exist_ok=True)

    jobs = []
    for g in groups:
        for i in range(n_subjects):
            jobs.append((g, i, seed * 1000 + i, tests, log_every, i < 2))

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    rows, traj, curves = [], [], []
    t0 = time.time()
    done = 0

    def collect(res):
        nonlocal done
        rows.append(res["row"])
        traj.extend(res["trajectory"])
        curves.extend(res["curves"])
        if "brain" in res:
            r = res["row"]
            safe = "".join(ch if ch.isalnum() else "_" for ch in r["group"])
            np.savez_compressed(out / "brains" / f"{safe}_s{r['subject']}.npz",
                                group=r["group"], **res["brain"])
        done += 1
        if progress:
            progress(done, len(jobs), res["row"]["group"])

    if workers == 1:
        for j in jobs:
            collect(_job(j))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_job, j) for j in jobs]
            for f in as_completed(futures):
                collect(f.result())

    order = {g.name: k for k, g in enumerate(groups)}
    df = pd.DataFrame(rows)
    df["group_order"] = df["group"].map(order)
    df.sort_values(["group_order", "subject"]).drop(columns="group_order").to_csv(out / "subjects.csv", index=False)
    pd.DataFrame(traj).to_csv(out / "development.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "learning_curves.csv", index=False)

    meta = {
        "design": design,
        "description": DESCRIPTIONS[design],
        "groups": [{"name": g.name, "factors": g.factors,
                    "phases": [{"env": asdict(p.env), "steps": p.steps} for p in g.phases]}
                   for g in groups],
        "n_subjects_per_group": n_subjects,
        "development_steps": steps,
        "seed": seed,
        "brain": asdict(groups[0].brain),
        "tests": asdict(tests),
        "enrichsim_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "date": datetime.now().isoformat(timespec="seconds"),
        "duration_s": round(time.time() - t0, 1),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
