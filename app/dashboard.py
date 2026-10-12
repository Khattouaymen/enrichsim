"""Tableau de bord EnrichSim pour la soutenance.

    streamlit run app/dashboard.py

Onglets : environnements (cages et labyrinthes), résultats (graphiques interactifs et
statistiques), cerveaux (connectivité des sujets enregistrés), lancement d'une expérience.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from enrichsim import stats as S  # noqa: E402
from enrichsim.config import CONDITIONS, BrainConfig, ablate, EC  # noqa: E402
from enrichsim.environment import MAZES, HomeCage, Maze  # noqa: E402
from enrichsim.experiment import DESCRIPTIONS, DESIGNS, run_experiment  # noqa: E402
from enrichsim.metrics import modality_slices  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
# 0 vide, 1 mur, 2 nourriture, 3 objet, 4 congénère, 5 sujet
CELL_COLORS = ["#f4f3ef", "#52514e", "#1baf7a", "#eda100", "#6250d6", "#e34948"]
CELL_NAMES = ["vide", "mur", "nourriture", "objet", "congénère", "sujet"]

st.set_page_config(page_title="EnrichSim", page_icon="🧠", layout="wide")


def colors_for(design: str, order: list[str]) -> dict[str, str]:
    pal = SEQ if design == "dose" else SERIES
    return {g: pal[i % len(pal)] for i, g in enumerate(order)}


def grid_figure(img: np.ndarray, title: str) -> go.Figure:
    scale = [x for i, c in enumerate(CELL_COLORS) for x in ([(i) / 6, c], [(i + 1) / 6, c])]
    names = np.vectorize(lambda v: CELL_NAMES[v])(img)
    fig = go.Figure(go.Heatmap(z=img, zmin=-0.5, zmax=5.5, colorscale=scale, showscale=False,
                               customdata=names, hovertemplate="%{customdata}<extra></extra>",
                               xgap=1, ygap=1))
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10),
                      yaxis=dict(autorange="reversed", visible=False, scaleanchor="x"),
                      xaxis=dict(visible=False), plot_bgcolor="white")
    return fig


def legend_html() -> str:
    items = "".join(f'<span style="display:inline-flex;align-items:center;margin-right:14px">'
                    f'<span style="width:12px;height:12px;border-radius:3px;background:{c};margin-right:5px">'
                    f"</span>{n}</span>" for c, n in zip(CELL_COLORS, CELL_NAMES))
    return f'<div style="font-size:0.9rem">{items}</div>'


def list_experiments(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted(p for p in base.iterdir() if (p / "subjects.csv").exists() and (p / "metadata.json").exists())


@st.cache_data(show_spinner=False)
def load(folder: str, mtime: float):
    f = Path(folder)
    df = pd.read_csv(f / "subjects.csv")
    meta = json.loads((f / "metadata.json").read_text(encoding="utf-8"))
    dev = pd.read_csv(f / "development.csv") if (f / "development.csv").exists() else pd.DataFrame()
    curves = pd.read_csv(f / "learning_curves.csv") if (f / "learning_curves.csv").exists() else pd.DataFrame()
    return df, meta, dev, curves


@st.cache_data(show_spinner="Analyses statistiques…")
def analyze(folder: str, mtime: float):
    R = S.analyze(folder, save=False)
    return {k: v for k, v in R.items() if k != "meta"}


# =========================================================================== barre latérale
st.sidebar.title("🧠 EnrichSim")
st.sidebar.caption("Enrichissement environnemental et développement d'un cerveau artificiel")
base = Path(st.sidebar.text_input("Dossier des résultats", str(ROOT / "results")))
experiments = list_experiments(base)

tab_env, tab_res, tab_brain, tab_run = st.tabs(
    ["Environnements", "Résultats", "Cerveaux", "Lancer une expérience"])

# =========================================================================== environnements
with tab_env:
    st.header("Les conditions d'élevage (Rosenzweig, Bennett & Diamond, 1972)")
    st.markdown(legend_html(), unsafe_allow_html=True)
    seed = st.number_input("Graine de la disposition", 0, 10_000, 1, key="cage_seed")
    cfg = BrainConfig()
    envs = dict(CONDITIONS)
    if st.checkbox("Montrer aussi les cages de l'expérience d'ablation"):
        for comp, lab in (("objects", "sans objets"), ("social", "sans social"), ("space", "sans espace")):
            envs[f"EC {lab}"] = ablate(EC, comp)
    cols = st.columns(3)
    for k, (name, env) in enumerate(envs.items()):
        cage = HomeCage(env, cfg, np.random.default_rng(seed))
        with cols[k % 3]:
            st.plotly_chart(grid_figure(cage.render(), f"{name} — {env.size}×{env.size}"),
                            width="stretch", key=f"cage{k}")
            st.caption(f"{env.n_objects} objets (parmi {env.object_pool} types), "
                       f"renouvellement {'toutes les ' + str(env.novelty_period) + ' étapes' if env.novelty_period else 'jamais'}, "
                       f"{env.n_conspecifics} congénères, {env.inner_walls} cloisons")
    st.subheader("Labyrinthes de Hebb-Williams (test)")
    cols = st.columns(len(MAZES))
    for p, c in enumerate(cols):
        m = Maze(p, cfg, np.random.default_rng(0))
        with c:
            st.plotly_chart(grid_figure(m.render(), f"Problème {p + 1}"),
                            width="stretch", key=f"maze{p}")

# =========================================================================== résultats
with tab_res:
    if not experiments:
        st.info(f"Aucun résultat dans {base}. Lancez une expérience (dernier onglet) ou la commande "
                "`python -m enrichsim run all --n 20 --steps 20000 --seed 2027 --out results`.")
    else:
        names = [p.name for p in experiments]
        choice = st.selectbox("Expérience", names, format_func=lambda n: f"{n} — "
                              + DESCRIPTIONS.get(n, n))
        folder = base / choice
        mtime = (folder / "subjects.csv").stat().st_mtime
        df, meta, dev, curves = load(str(folder), mtime)
        R = analyze(str(folder), mtime)
        order, design = R["order"], R["design"]
        colors = colors_for(design, order)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Groupes", len(order))
        c2.metric("Sujets", len(df))
        c3.metric("Élevage (étapes)", f"{meta['development_steps']:,}".replace(",", " "))
        c4.metric("Graine", meta["seed"])

        variables = [v for v in S.VARIABLES if v in df]
        var = st.selectbox("Variable", variables, format_func=S.label,
                           index=variables.index("brain_n_synapses") if "brain_n_synapses" in variables else 0)
        left, right = st.columns([3, 2])
        with left:
            fig = go.Figure()
            for g in order:
                x = df.loc[df.group == g, var]
                fig.add_trace(go.Box(y=x, name=g, marker_color=colors[g], boxpoints="all", jitter=0.4,
                                     pointpos=0, line_width=1.5, fillcolor="rgba(0,0,0,0)",
                                     hovertemplate=f"{g}<br>%{{y:.3f}}<extra></extra>"))
            fig.update_layout(title=S.label(var), showlegend=False, height=420,
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")
        with right:
            om = R["omnibus"]
            row = om[om.variable == var]
            if len(row):
                r = row.iloc[0]
                st.markdown(f"**{r.test}** — {r.apa}")
                st.markdown(f"p corrigé (Holm, famille *{r.family}*) : **{S.fmt_p(r.p_holm)}**")
                ph = R["posthoc"]
                ph = ph[ph.variable == var][["A", "B", "diff", "p", "g"]].rename(
                    columns={"A": "Groupe A", "B": "Groupe B", "diff": "Différence", "g": "g de Hedges"})
                ph["p"] = ph["p"].map(lambda p: S.fmt_p(p).replace("p = ", "").replace("p ", ""))
                st.dataframe(ph.round(3), hide_index=True, width="stretch")
            d = R["descriptives"]
            st.dataframe(d[d.variable == var][["group", "n", "M", "SD", "CI_low", "CI_high"]].round(3)
                         .rename(columns={"group": "Groupe", "SD": "ET", "CI_low": "IC95 bas",
                                          "CI_high": "IC95 haut"}), hide_index=True, width="stretch")

        st.subheader("Tous les tests globaux")
        show = R["omnibus"][["family", "label", "test", "apa", "p_holm", "friedman_p"]].copy()
        show["p_holm"] = show["p_holm"].map(lambda p: S.fmt_p(p).replace("p = ", "").replace("p ", ""))
        show["friedman_p"] = show["friedman_p"].map(lambda p: S.fmt_p(p).replace("p = ", "").replace("p ", ""))
        st.dataframe(show.rename(columns={"family": "Famille", "label": "Variable", "test": "Test",
                                          "apa": "Résultat (APA)", "p_holm": "p (Holm)",
                                          "friedman_p": "p Friedman"}),
                     hide_index=True, width="stretch")

        if not dev.empty:
            st.subheader("Trajectoires développementales")
            dvars = [c for c in ("n_synapses", "n_neurons", "fan_in_mean", "coverage_objects",
                                 "pattern_separation", "dimensionality", "prediction_error") if c in dev]
            dv = st.selectbox("Mesure", dvars, format_func=lambda c: S.label("brain_" + c))
            fig = go.Figure()
            for g in order:
                agg = dev[dev.group == g].groupby("age")[dv].agg(["mean", "sem"]).reset_index()
                fig.add_trace(go.Scatter(x=agg.age, y=agg["mean"], name=g, mode="lines",
                                         line=dict(color=colors[g], width=2),
                                         error_y=dict(type="data", array=1.96 * agg["sem"], visible=True,
                                                      thickness=1, width=0)))
            fig.update_layout(xaxis_title="Âge (étapes)", yaxis_title=S.label("brain_" + dv), height=400,
                              hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")

        if not curves.empty:
            st.subheader("Courbes d'apprentissage")
            task = st.radio("Tâche", ["maze", "reversal"], horizontal=True,
                            format_func=lambda t: "Hebb-Williams" if t == "maze" else "Apprentissage inversé")
            cv = curves[curves.task == task].groupby(["group", "subject", "trial"])["errors"].mean().reset_index()
            fig = go.Figure()
            for g in order:
                agg = cv[cv.group == g].groupby("trial")["errors"].agg(["mean", "sem"]).reset_index()
                fig.add_trace(go.Scatter(x=agg.trial, y=agg["mean"], name=g, mode="lines+markers",
                                         line=dict(color=colors[g], width=2), marker=dict(size=8)))
            fig.update_layout(xaxis_title="Essai", yaxis_title="Erreurs", height=380, hovermode="x unified",
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
            m = R["mixed"].get(task, {})
            if "apa_interaction" in m:
                st.markdown(f"Modèle mixte, interaction groupe × essai : **{m['apa_interaction']}**")

        rep = folder / "rapport.html"
        if rep.exists():
            with st.expander("Rapport complet (HTML)"):
                st.download_button("Télécharger le rapport", rep.read_bytes(), file_name=f"rapport_{choice}.html",
                                   mime="text/html")
                components.html(rep.read_text(encoding="utf-8"), height=900, scrolling=True)

# =========================================================================== cerveaux
with tab_brain:
    files = sorted(base.glob("*/brains/*.npz")) if base.exists() else []
    if not files:
        st.info("Aucun cerveau enregistré (les 2 premiers sujets de chaque groupe sont sauvegardés).")
    else:
        st.header("Connectivité des cerveaux en fin d'élevage")
        st.caption("Chaque ligne est un neurone cortical, chaque colonne une entrée sensorielle ; "
                   "un point coloré est une synapse. Comparer un sujet IC et un sujet EC : même cerveau à la "
                   "naissance (plan apparié), environnement différent.")
        cfg = BrainConfig()
        mods = modality_slices(cfg)
        order_in = np.concatenate([mods[k] for k in ("walls", "food", "social", "objects", "place")])
        bounds = np.cumsum([len(mods[k]) for k in ("walls", "food", "social", "objects", "place")])
        labels = [f.parent.parent.name + " / " + f.stem for f in files]
        default = [lab for lab in labels if lab.endswith(("IC_s0", "EC_s0"))][:2] or labels[:2]
        pick = st.multiselect("Sujets", labels, default=default, max_selections=3)
        cols = st.columns(max(1, len(pick)))
        for c, lab in zip(cols, pick):
            d = np.load(files[labels.index(lab)])
            M, W, alive = d["M"], d["W1"], d["alive"]
            Wa = np.abs(W[alive][:, order_in]) * M[alive][:, order_in]
            with c:
                st.markdown(f"**{lab}** — {int(alive.sum())} neurones, {int(M[alive].sum())} synapses")
                fig = go.Figure(go.Heatmap(z=Wa, colorscale=[[0, "#ffffff"], [0.001, "#cde2fb"], [1, "#104281"]],
                                           showscale=False,
                                           hovertemplate="neurone %{y}<br>entrée %{x}<br>|w| = %{z:.3f}<extra></extra>"))
                for b in bounds[:-1]:
                    fig.add_vline(x=b - 0.5, line_color="#c3c2b7", line_width=1)
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=30),
                                  xaxis=dict(title="entrées : murs | nourriture | social | objets | lieu",
                                             showticklabels=False),
                                  yaxis=dict(title="neurones", showticklabels=False))
                st.plotly_chart(fig, width="stretch", key=f"brain{lab}")
                fr = {"walls": "murs", "food": "nourriture", "social": "congénères", "objects": "objets",
                      "place": "cellules de lieu"}
                cov = {fr[k]: float(M[alive][:, v].sum() / max(1, alive.sum() * len(v))) for k, v in mods.items()}
                st.dataframe(pd.DataFrame({"Modalité": list(cov), "Couverture": [round(x, 3) for x in cov.values()]}),
                             hide_index=True, width="stretch")

# =========================================================================== lancement
with tab_run:
    st.header("Lancer une expérience")
    st.caption("Pour une démonstration en direct, garder un petit effectif (3 sujets, 3 000 étapes : "
               "environ une minute). L'expérience du mémoire : 20 sujets, 20 000 étapes, graine 2027.")
    c1, c2, c3, c4 = st.columns(4)
    design = c1.selectbox("Plan", list(DESIGNS), format_func=lambda d: f"{d} — {DESCRIPTIONS[d]}")
    n = c2.number_input("Sujets par groupe", 2, 100, 3)
    steps = c3.number_input("Durée de l'élevage (étapes)", 500, 100_000, 3000, step=500)
    seed = c4.number_input("Graine", 0, 100_000, 2027)
    out = st.text_input("Dossier de sortie", str(ROOT / "demo" / design))
    if st.button("Lancer", type="primary"):
        bar = st.progress(0.0, text="Initialisation…")

        def progress(done, total, group):
            bar.progress(done / total, text=f"{done}/{total} sujets — dernier terminé : {group}")

        folder = run_experiment(design, n_subjects=int(n), steps=int(steps), out_dir=out, seed=int(seed),
                                progress=progress)
        bar.progress(1.0, text="Analyse et rapport…")
        from enrichsim.report import build_report
        path = build_report(folder)
        st.success(f"Terminé : {folder}. Rapport : {path}")
        st.download_button("Télécharger le rapport", path.read_bytes(), file_name=f"rapport_{design}.html",
                           mime="text/html")
        components.html(path.read_text(encoding="utf-8"), height=900, scrolling=True)
