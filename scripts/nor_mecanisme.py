"""Analyse exploratoire : pourquoi la discrimination neuronale ne devient-elle pas comportementale ?

Rejoue l'élevage et les deux premiers tests (champ ouvert, reconnaissance d'objet) de
8 sujets IC et 8 sujets EC de l'expérience principale (mêmes graines, donc mêmes sujets) et
enregistre, à chaque contact avec un objet pendant la phase de test du NOR :
la nouveauté perçue, le bonus d'investigation qui en découle et s'il atteint son plafond.

Vérification intégrée : les indices de discrimination recalculés doivent être identiques à
ceux de results/main/subjects.csv.

    python scripts/nor_mecanisme.py        (environ 1 minute sur 4 cœurs)
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from enrichsim.brain import OBJ_SCALE  # noqa: E402
from enrichsim.config import CONDITIONS, BrainConfig, Phase, TestConfig  # noqa: E402
from enrichsim.environment import NOVEL_OBJECT_BASE, ObjectArena  # noqa: E402
from enrichsim.subject import Subject  # noqa: E402
from enrichsim.tasks import open_field  # noqa: E402

SEED, STEPS, N = 2027, 20_000, 8
OUT = ROOT / "results" / "main" / "exploratoire_nor"


def replay_subject(args: tuple[str, int]) -> tuple[list[dict], dict]:
    cond, i = args
    cfg, tests = BrainConfig(), TestConfig()
    s = Subject(i, cfg, SEED * 1000 + i)
    s.develop([Phase(CONDITIONS[cond], STEPS)], log_every=max(500, STEPS // 20))
    open_field(s, tests)
    rows, dis = [], []
    size = 7
    for k in range(tests.nor_sessions):           # même déroulement que tasks._nor_session
        fam, nov = NOVEL_OBJECT_BASE + 2 * k + 1, NOVEL_OBJECT_BASE + 2 * k + 2
        novel_right = (s.sid + k) % 2 == 0
        arena = ObjectArena(size, cfg, s.rng, fam, fam)
        s.run(arena, tests.nor_familiarization)
        if novel_right:
            arena.set_objects(fam, nov)
        else:
            arena.set_objects(nov, fam)
        arena.agent = np.array([size - 1, size // 2])
        pos_nov = arena.right if novel_right else arena.left
        pos_fam = arena.left if novel_right else arena.right
        t = {"nov": 0, "fam": 0}

        def on_step(world, info, out):
            for key, pos in (("nov", pos_nov), ("fam", pos_fam)):
                if info.get("object_pos") == pos:
                    t[key] += 1
                    ratio = out["object_novelty"] / OBJ_SCALE
                    rows.append({"group": cond, "subject": i, "session": k, "object": key,
                                 "novelty": out["object_novelty"], "capped": ratio >= 2.0,
                                 "bonus": cfg.investigation * min(ratio, 2.0)})

        s.run(arena, tests.nor_test, on_step=on_step)
        total = t["nov"] + t["fam"]
        dis.append((t["nov"] - t["fam"]) / total if total >= 5 else np.nan)
    return rows, {"group": cond, "subject": i, "nor_di": float(np.nanmean(dis))}


def main() -> None:
    jobs = [(c, i) for c in ("IC", "EC") for i in range(N)]
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(replay_subject, jobs))
    contacts = pd.DataFrame([r for rows, _ in res for r in rows])
    di = pd.DataFrame([d for _, d in res])

    ref = pd.read_csv(ROOT / "results" / "main" / "subjects.csv").set_index(["group", "subject"])["nor_di"]
    assert all(abs(ref.loc[(r.group, r.subject)] - r.nor_di) < 1e-12 for r in di.itertuples()), \
        "les indices recalculés diffèrent de results/main/subjects.csv"

    OUT.mkdir(parents=True, exist_ok=True)
    contacts.to_csv(OUT / "contacts.csv", index=False)
    summary = contacts.groupby(["group", "object"]).agg(
        contacts=("novelty", "size"), nouveaute=("novelty", "mean"),
        part_plafond=("capped", "mean"), bonus=("bonus", "mean")).reset_index()
    per_subject = contacts.groupby(["group", "subject"]).size()
    p = (contacts.object == "nov").mean()
    sd_sampling = 2 * np.sqrt(p * (1 - p) / per_subject.mean())
    summary.to_csv(OUT / "resume.csv", index=False)
    lines = [
        summary.round(3).to_string(index=False),
        f"contacts par session de test : {contacts.groupby(['group', 'subject', 'session']).size().mean():.1f}",
        f"contacts par sujet (4 sessions) : {per_subject.mean():.1f}",
        f"écart-type de l'indice dû au seul échantillonnage : {sd_sampling:.3f}",
        "indices de discrimination recalculés identiques à results/main/subjects.csv : oui",
    ]
    (OUT / "resume.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
