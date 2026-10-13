"""Protocoles expérimentaux et exécution parallèle.

Cinq expériences sont définies :

1. `main`     — IC vs SC vs EC (réplique de Rosenzweig et al., 1962-1972).
2. `dose`     — relation dose-réponse : 5 niveaux d'enrichissement de 0 à 1.
3. `timing`   — période critique et réversibilité : plan factoriel 2 x 2
                (enrichissement précoce oui/non) x (enrichissement tardif oui/non),
                avec une plasticité qui décroît avec l'âge.
4. `ablation` — quelle composante de l'enrichissement compte ? (objets, nouveauté,
                social, espace), chacune retirée séparément de EC.
5. `control`  — contrôle exploratoire, ajouté après l'expérience finale : même élevage
                que `main` (mêmes graines, donc mêmes cerveaux en fin d'élevage), mais
                l'architecture du cerveau est figée pendant les tests.

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


def design_control(steps: int, brain: BrainConfig) -> list[Group]:
    return design_main(steps, brain)


DESIGNS: dict[str, Callable[[int, BrainConfig], list[Group]]] = {
    "main": design_main,
    "dose": design_dose,
    "timing": design_timing,
    "ablation": design_ablation,
    "control": design_control,
}

# réglages de la batterie de tests propres à un plan
DESIGN_TESTS: dict[str, dict] = {"control": {"freeze_structure": True}}

DESCRIPTIONS = {
    "main": "Effet de l'environnement d'élevage (IC / SC / EC)",
    "dose": "Relation dose-réponse de l'enrichissement",
    "timing": "Période critique : enrichissement précoce vs tardif (2 x 2)",
    "ablation": "Contribution de chaque composante de l'enrichissement",
    "control": "Contrôle : IC / SC / EC, architecture du cerveau figée pendant les tests",
}


# --------------------------------------------------------------------------- exécution d'un sujet

def run_subject(group: Group, index: int, seed: int, tests: TestConfig,
                log_every: int, keep_brain: bool) -> dict:
    t0 = time.time()
    subject = Subject(index, group.brain, seed)
    trajectory = subject.develop(group.phases, log_every=log_every)
    anatomy = {f"brain_{k}": v for k, v in trajectory[-1].items()
               if k not in ("age", "phase", "env")}
    brain_end = subject.brain.state() if keep_brain else None   # cerveau en fin d'élevage
    behaviour, curves, stages = test_battery(subject, tests)
    row = {"group": group.name, "subject": index, "seed": seed, **group.factors,
           **anatomy, **behaviour, "runtime_s": round(time.time() - t0, 2)}
    for r in (*trajectory, *curves, *stages):
        r.update(group=group.name, subject=index)
    out = {"row": row, "trajectory": trajectory, "curves": curves, "stages": stages}
    if keep_brain:
        out["brain"] = brain_end
        out["brain_post"] = subject.brain.state()                # cerveau après les tests
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
      subjects.csv          une ligne par sujet (anatomie en fin d'élevage + comportement)
      development.csv       trajectoires développementales pendant l'élevage
      test_development.csv  état du cerveau avant la batterie de tests et après chaque test
      learning_curves.csv   performances essai par essai (labyrinthe, inversion)
      brains/*.npz          cerveaux des 2 premiers sujets de chaque groupe, en fin d'élevage
                            (<groupe>_s<i>.npz) et après les tests (<groupe>_s<i>_apres_tests.npz)
      metadata.json         paramètres complets (reproductibilité)

    Toutes les tables sont triées (groupe, sujet, ...) : à graine égale, les fichiers sont
    identiques d'une exécution à l'autre, à la durée de calcul (`runtime_s`) près.
    """
    tests = replace(tests or TestConfig(), **DESIGN_TESTS.get(design, {}))
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
    rows, traj, curves, stages = [], [], [], []
    t0 = time.time()
    done = 0

    def collect(res):
        nonlocal done
        rows.append(res["row"])
        traj.extend(res["trajectory"])
        curves.extend(res["curves"])
        stages.extend(res["stages"])
        if "brain" in res:
            r = res["row"]
            safe = "".join(ch if ch.isalnum() else "_" for ch in r["group"])
            np.savez_compressed(out / "brains" / f"{safe}_s{r['subject']}.npz",
                                group=r["group"], stage="fin d'élevage", **res["brain"])
            np.savez_compressed(out / "brains" / f"{safe}_s{r['subject']}_apres_tests.npz",
                                group=r["group"], stage="après les tests", **res["brain_post"])
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

    def save(records: list[dict], name: str, keys: list[str]) -> None:
        df = pd.DataFrame(records)
        df["group_order"] = df["group"].map(order)
        df = df.sort_values(["group_order", "subject", *keys], kind="stable")
        df.drop(columns="group_order").to_csv(out / name, index=False)

    save(rows, "subjects.csv", [])
    save(traj, "development.csv", ["age"])
    save(stages, "test_development.csv", ["stage"])
    save(curves, "learning_curves.csv", ["task", "problem", "trial"])

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
