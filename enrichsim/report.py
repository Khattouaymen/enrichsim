"""Rapport HTML autonome (en français) à partir d'un dossier de résultats.

    python -m enrichsim report results/main

Le rapport contient : rappel du plan, statistiques descriptives, tests globaux au format
APA, comparaisons post-hoc, courbes d'apprentissage (modèle mixte), analyses propres au
plan (dose, période critique, ablation) et trajectoires développementales. Les figures
sont intégrées en base64 : le fichier HTML se suffit à lui-même (envoi par courriel,
impression en PDF depuis le navigateur).
"""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import stats as S  # noqa: E402

# --------------------------------------------------------------------------- style des figures
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # ordinal clair -> foncé
TEXT, TEXT2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 9.5,
    "axes.titleweight": "bold", "axes.edgecolor": AXIS, "axes.labelcolor": TEXT2,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
})


def group_colors(design: str, order: list[str]) -> dict[str, str]:
    if design == "dose":
        return {g: SEQ[i % len(SEQ)] for i, g in enumerate(order)}
    return {g: SERIES[i % len(SERIES)] for i, g in enumerate(order)}


def fig_to_html(fig, alt: str) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'


def strip_panels(df: pd.DataFrame, variables: list[str], order: list[str],
                 colors: dict[str, str], ncol: int = 4) -> str:
    """Un panneau par variable : sujets individuels + moyenne ± IC 95 %."""
    n = len(variables)
    nrow = int(np.ceil(n / ncol))
    wide = len(order) > 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * (2.6 if not wide else 3.2), nrow * 2.5),
                             squeeze=False)
    rng = np.random.default_rng(0)
    for ax, v in zip(axes.ravel(), variables):
        for i, g in enumerate(order):
            x = S._clean(df.loc[df.group == g, v])
            if not len(x):
                continue
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(x)), x, s=10, color=colors[g], alpha=0.35,
                       linewidths=0)
            m = x.mean()
            ci = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0
            ax.errorbar(i, m, yerr=ci, fmt="o", color=TEXT, ms=5, capsize=3, lw=1.4,
                        mec="white", mew=1.2, zorder=5)
        ax.set_title(S.label(v), loc="left", wrap=True)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=35 if wide else 0, ha="right" if wide else "center")
        ax.grid(axis="x", visible=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig_to_html(fig, "Distribution par groupe : sujets individuels, moyenne et IC 95 %")


def learning_figure(curves: pd.DataFrame, order: list[str], colors: dict[str, str]) -> str:
    tasks = [t for t in ("maze", "reversal") if t in set(curves["task"])]
    titles = {"maze": "Hebb-Williams (moyenne des problèmes)", "reversal": "Apprentissage inversé"}
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 3.2), squeeze=False)
    for ax, t in zip(axes.ravel(), tasks):
        d = curves[curves.task == t].groupby(["group", "subject", "trial"])["errors"].mean().reset_index()
        for g in order:
            dg = d[d.group == g].groupby("trial")["errors"]
            m, se = dg.mean(), dg.sem()
            ax.plot(m.index, m.values, color=colors[g], label=g, marker="o", ms=4,
                    mec="white", mew=1)
            ax.fill_between(m.index, m - 1.96 * se, m + 1.96 * se, color=colors[g], alpha=0.10, lw=0)
        ax.set_title(titles[t], loc="left")
        ax.set_xlabel("Essai")
        ax.set_ylabel("Erreurs (moyenne, IC 95 %)")
        ax.grid(axis="x", visible=False)
    axes.ravel()[0].legend(fontsize=8, ncol=1 if len(order) < 5 else 2)
    fig.tight_layout()
    return fig_to_html(fig, "Courbes d'apprentissage par groupe")


DEV_VARS = ["n_synapses", "n_neurons", "coverage_objects", "pattern_separation"]
DEV_LABELS = {"n_synapses": "Synapses", "n_neurons": "Neurones",
              "coverage_objects": "Territoire « objets »", "pattern_separation": "Séparation de patterns"}


def development_figure(dev: pd.DataFrame, order: list[str], colors: dict[str, str]) -> str:
    vars_ = [v for v in DEV_VARS if v in dev]
    fig, axes = plt.subplots(1, len(vars_), figsize=(3.3 * len(vars_), 2.9), squeeze=False)
    for ax, v in zip(axes.ravel(), vars_):
        for g in order:
            dg = dev[dev.group == g].groupby("age")[v]
            m, se = dg.mean(), dg.sem()
            ax.plot(m.index, m.values, color=colors[g], label=g)
            ax.fill_between(m.index, m - 1.96 * se, m + 1.96 * se, color=colors[g], alpha=0.10, lw=0)
        ax.set_title(DEV_LABELS[v], loc="left")
        ax.set_xlabel("Âge (étapes)")
        ax.grid(axis="x", visible=False)
        if "phase" in dev and dev["phase"].max() > 0:
            switch = dev.loc[dev.phase == 1, "age"].min()
            ax.axvline(switch, color=AXIS, lw=1)
    axes.ravel()[0].legend(fontsize=8, ncol=1 if len(order) < 5 else 2)
    fig.tight_layout()
    return fig_to_html(fig, "Trajectoires développementales")


STAGE_FIG_VARS = {"n_synapses": "Synapses", "coverage_objects": "Territoire « objets »"}


def test_stage_figure(stages: pd.DataFrame, order: list[str], colors: dict[str, str]) -> str:
    """Anatomie avant la batterie de tests puis après chaque test (moyenne et IC 95 %)."""
    labels = stages.drop_duplicates("stage").sort_values("stage")["stage_label"].tolist()
    fig, axes = plt.subplots(1, len(STAGE_FIG_VARS), figsize=(5 * len(STAGE_FIG_VARS), 3.3), squeeze=False)
    for ax, (v, title) in zip(axes.ravel(), STAGE_FIG_VARS.items()):
        for g in order:
            dg = stages[stages.group == g].groupby("stage")[v]
            m, se = dg.mean(), dg.sem()
            ax.plot(m.index, m.values, color=colors[g], label=g, marker="o", ms=4, mec="white", mew=1)
            ax.fill_between(m.index, m - 1.96 * se, m + 1.96 * se, color=colors[g], alpha=0.10, lw=0)
        ax.set_title(title, loc="left")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(["avant les tests"] + [f"après : {lab}" for lab in labels[1:]],
                           rotation=25, ha="right")
        ax.grid(axis="x", visible=False)
    axes.ravel()[0].legend(fontsize=8, ncol=1 if len(order) < 5 else 2)
    fig.tight_layout()
    return fig_to_html(fig, "Anatomie du cerveau pendant la batterie de tests")


def dose_figure(df: pd.DataFrame, variables: list[str]) -> str:
    fig, axes = plt.subplots(1, len(variables), figsize=(3.2 * len(variables), 2.8), squeeze=False)
    for ax, v in zip(axes.ravel(), variables):
        g = df.groupby("level")[v]
        m, se = g.mean(), g.sem()
        ax.scatter(df["level"] + np.random.default_rng(0).uniform(-0.02, 0.02, len(df)), df[v],
                   s=8, color=SERIES[0], alpha=0.3, linewidths=0)
        ax.errorbar(m.index, m.values, yerr=1.96 * se, color=SERIES[0], marker="o", ms=5,
                    mec="white", mew=1.2, capsize=3)
        ax.set_title(S.label(v), loc="left", wrap=True)
        ax.set_xlabel("Niveau d'enrichissement (0 = IC, 1 = EC)")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig_to_html(fig, "Relation dose-réponse")


def timing_figure(df: pd.DataFrame, variables: list[str]) -> str:
    fig, axes = plt.subplots(1, len(variables), figsize=(3.2 * len(variables), 2.9), squeeze=False)
    for ax, v in zip(axes.ravel(), variables):
        for k, early in enumerate((0, 1)):
            d = df[df.early == early].groupby("late")[v]
            m, se = d.mean(), d.sem()
            ax.errorbar(np.array(m.index) + (k - 0.5) * 0.04, m.values, yerr=1.96 * se,
                        color=SERIES[k], marker="o", ms=5, mec="white", mew=1.2, capsize=3,
                        label=f"1re moitié : {'EC' if early else 'IC'}")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["IC", "EC"])
        ax.set_xlabel("2e moitié de l'élevage")
        ax.set_title(S.label(v), loc="left", wrap=True)
        ax.grid(axis="x", visible=False)
    axes.ravel()[0].legend(fontsize=8)
    fig.tight_layout()
    return fig_to_html(fig, "Plan 2 x 2 : graphique d'interaction")


def ablation_figure(abl: pd.DataFrame, variables: list[str]) -> str:
    d = abl[abl.variable.isin(variables) & (abl.group != "IC") & abl.pct_effect_lost.notna()]
    variables = [v for v in variables if v in set(d.variable)]
    if not variables:
        return "<p>Aucune variable ne distingue EC de IC : pourcentage non calculable.</p>"
    groups = list(dict.fromkeys(d.group))
    fig, ax = plt.subplots(figsize=(7.5, 0.5 + 0.45 * len(variables) * len(groups) / 2))
    y, ticks, labels = 0, [], []
    for v in variables:
        dv = d[d.variable == v]
        for k, g in enumerate(groups):
            row = dv[dv.group == g]
            if row.empty:
                continue
            val = float(row.pct_effect_lost.iloc[0])
            if not np.isfinite(val):
                y += 1
                continue
            ax.barh(y, val, height=0.7, color=SERIES[k % len(SERIES)], label=g if v == variables[0] else None)
            y += 1
        ticks.append(y - len(groups) / 2 - 0.5)
        labels.append(S.label(v))
        y += 0.8
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color=AXIS, lw=1)
    ax.axvline(100, color=AXIS, lw=1)
    ax.set_xlabel("Part de l'écart EC – IC perdue en retirant la composante (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig_to_html(fig, "Ablation : part de l'effet perdue")


# --------------------------------------------------------------------------- tableaux HTML

def table(df: pd.DataFrame, cols: dict[str, str], fmt: dict[str, callable] | None = None) -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{html.escape(h)}</th>" for h in cols.values())
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, "")
            if c in fmt:
                s = fmt[c](v)
            elif isinstance(v, (float, np.floating)):
                s = S.fmt_num(float(v))
            else:
                s = str(v)
            num = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
            cells.append(f'<td class="{"num" if num else ""}">{html.escape(s)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def pfmt(p) -> str:
    return S.fmt_p(float(p)).replace("p = ", "").replace("p ", "") if pd.notna(p) else "n.d."


def sig(p) -> str:
    return "" if pd.isna(p) else "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def descriptives_table(desc: pd.DataFrame, order: list[str]) -> str:
    rows = []
    for v, dv in desc.groupby("variable", sort=False):
        r = {"Variable": S.label(v)}
        for g in order:
            x = dv[dv.group == g]
            if len(x):
                x = x.iloc[0]
                r[g] = f"{S.fmt_num(x.M)} ({S.fmt_num(x.SD)})"
        rows.append(r)
    t = pd.DataFrame(rows)
    return table(t, {c: c for c in t.columns})


def omnibus_table(omni: pd.DataFrame) -> str:
    t = omni.copy()
    t["sig"] = t["p_holm"].map(sig)
    return table(t, {"label": "Variable", "test": "Test", "apa": "Résultat (APA)",
                     "p_holm": "p (Holm)", "sig": "", "friedman_p": "p Friedman (apparié)"},
                 {"p_holm": pfmt, "friedman_p": pfmt})


def posthoc_table(post: pd.DataFrame, variables: list[str]) -> str:
    t = post[post.variable.isin(variables)].copy()
    t["pair"] = t["A"] + " – " + t["B"]
    t["sig"] = t["p"].map(sig)
    return table(t, {"label": "Variable", "method": "Méthode", "pair": "Comparaison",
                     "diff": "Différence", "p": "p", "sig": "", "g": "g de Hedges"},
                 {"p": pfmt, "g": lambda x: S.fmt_num(x)})


# --------------------------------------------------------------------------- texte d'interprétation

def magnitude(g: float) -> str:
    a = abs(g)
    return "négligeable" if a < .2 else "petit" if a < .5 else "moyen" if a < .8 else "grand"


def summary_sentences(omni: pd.DataFrame, post: pd.DataFrame) -> list[str]:
    out = []
    for _, r in omni.iterrows():
        if pd.isna(r.p_holm):
            continue
        if r.p_holm < S.ALPHA:
            ph = post[(post.variable == r.variable) & (post.p < S.ALPHA)]
            pairs = ", ".join(f"{a} vs {b} (g = {S.fmt_num(g)}, effet {magnitude(g)})"
                              for a, b, g in zip(ph.A, ph.B, ph.g))
            out.append(f"<b>{html.escape(r.label)}</b> : effet significatif du groupe, {html.escape(r.apa)}"
                       f" (p corrigé de Holm {pfmt(r.p_holm)})."
                       + (f" Différences post-hoc significatives : {html.escape(pairs)}." if pairs else ""))
        else:
            out.append(f"<b>{html.escape(r.label)}</b> : pas d'effet significatif après correction "
                       f"({html.escape(r.apa)} ; p de Holm {pfmt(r.p_holm)}).")
    return out


# --------------------------------------------------------------------------- assemblage

CSS = """
:root{--bg:#fcfcfb;--fg:#0b0b0b;--fg2:#52514e;--line:#e1e0d9;--accent:#2a78d6;--card:#ffffff}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;--line:#2c2c2a;--accent:#3987e5;--card:#f7f7f5}}
:root[data-theme="dark"]{--bg:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;--line:#2c2c2a;--accent:#3987e5;--card:#f7f7f5}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:1.7rem;margin:0 0 4px}h2{font-size:1.25rem;margin:40px 0 8px;padding-top:12px;border-top:1px solid var(--line)}
h3{font-size:1.05rem;margin:24px 0 6px}
.sub{color:var(--fg2);margin:0 0 20px}
.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px 16px;margin:16px 0;padding:12px 16px;border:1px solid var(--line);border-radius:8px}
.meta div span{display:block;color:var(--fg2);font-size:.8rem}
.tw{overflow-x:auto;margin:8px 0 16px}
table{border-collapse:collapse;width:100%;font-size:.85rem;font-variant-numeric:tabular-nums}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--fg2);font-weight:600}
td.num{text-align:right;white-space:nowrap}
figure{margin:12px 0 20px;background:var(--card);border-radius:8px;padding:8px}
figure img{max-width:100%;height:auto;display:block}
figcaption{color:#52514e;font-size:.82rem;padding:6px 4px 2px}
.note{border-left:3px solid var(--accent);padding:8px 12px;margin:12px 0;color:var(--fg2)}
ul.res li{margin:4px 0}
@media print{body{background:#fff;color:#000}h2{page-break-before:auto}figure{page-break-inside:avoid}}
"""


def build_report(folder: str | Path) -> Path:
    folder = Path(folder)
    R = S.analyze(folder, save=True)
    meta, order, design = R["meta"], R["order"], R["design"]
    df = pd.read_csv(folder / "subjects.csv")
    dev = pd.read_csv(folder / "development.csv") if (folder / "development.csv").exists() else pd.DataFrame()
    curves = pd.read_csv(folder / "learning_curves.csv") if (folder / "learning_curves.csv").exists() else pd.DataFrame()
    colors = group_colors(design, order)
    anat, behav = S.family_vars(df, "anatomie"), S.family_vars(df, "comportement")
    omni, post = R["omnibus"], R["posthoc"]
    n_total = len(df)

    parts: list[str] = []
    add = parts.append
    add(f"<h1>EnrichSim — {html.escape(meta['description'])}</h1>")
    add('<p class="sub">Rapport généré automatiquement. Données simulées : il ne s\'agit pas '
        "d'une étude sur l'animal, mais d'un modèle computationnel dont les paramètres ont été "
        "fixés avant l'expérience."
        + (" Plan de contrôle exploratoire, ajouté après l'expérience finale." if design == "control" else "")
        + "</p>")
    add('<div class="meta">'
        f"<div><span>Plan</span>{html.escape(design)}</div>"
        f"<div><span>Groupes</span>{len(order)}</div>"
        f"<div><span>Sujets par groupe</span>{meta['n_subjects_per_group']}</div>"
        f"<div><span>Sujets au total</span>{n_total}</div>"
        f"<div><span>Durée de l'élevage</span>{meta['development_steps']:,} étapes</div>".replace(",", " ")
        + f"<div><span>Graine aléatoire</span>{meta['seed']}</div>"
        f"<div><span>Date</span>{html.escape(meta['date'][:10])}</div>"
        f"<div><span>Version</span>EnrichSim {html.escape(meta['enrichsim_version'])}</div>"
        "</div>")

    # ---------------------------------------------------------------- méthode
    add("<h2>1. Méthode</h2>")
    rows = []
    for g in meta["groups"]:
        ph = " → ".join(f"{p['env']['name']} ({p['steps']} ét.)" for p in g["phases"])
        e = g["phases"][-1]["env"]
        rows.append({"Groupe": g["name"], "Élevage": ph, "Taille": e["size"], "Objets": e["n_objects"],
                     "Nouveauté (période)": str(e["novelty_period"] or "—"), "Congénères": e["n_conspecifics"],
                     "Cloisons": e["inner_walls"]})
    t = pd.DataFrame(rows)
    add(table(t, {c: c for c in t.columns}))
    add('<p class="note">Plan apparié (« split-litter ») : le sujet n°i de chaque groupe démarre '
        "avec le même cerveau initial. Les tests globaux traitent néanmoins les groupes comme "
        "indépendants (analyse conservatrice) ; le test de Friedman, qui exploite l'appariement, "
        "est donné en complément. Le choix du test (ANOVA, Welch ou Kruskal-Wallis) dépend des "
        "tests de Shapiro-Wilk et de Levene (α = .05). Les p globaux sont corrigés par la méthode "
        "de Holm au sein de chaque famille (anatomie, comportement).</p>")

    # ---------------------------------------------------------------- anatomie
    for num, (fam, vars_, title) in enumerate(((("anatomie", anat, "Structure et représentations neuronales")),
                                               (("comportement", behav, "Comportement"))), start=2):
        add(f"<h2>{num}. {title}</h2>")
        add(f"<figure>{strip_panels(df, vars_, order, colors)}<figcaption>Figure {num}. Points : sujets "
            "individuels ; marqueur noir : moyenne et IC 95 %.</figcaption></figure>")
        add("<h3>Statistiques descriptives — M (ET)</h3>")
        add(descriptives_table(R["descriptives"][R["descriptives"].variable.isin(vars_)], order))
        o = omni[omni.family == fam] if not omni.empty else omni
        if not o.empty:
            add("<h3>Tests globaux</h3>")
            add(omnibus_table(o))
            add("<h3>Synthèse</h3><ul class='res'>" + "".join(f"<li>{s}</li>" for s in summary_sentences(o, post))
                + "</ul>")
            add("<details><summary>Comparaisons post-hoc détaillées</summary>")
            add(posthoc_table(post, vars_))
            add("</details>")

    # ---------------------------------------------------------------- NOR vs 0
    nz = R.get("nor_vs_zero")
    if nz is not None and not nz.empty:
        add("<h3>Préférence pour l'objet nouveau (indice comparé à 0)</h3>")
        nz = nz.assign(label=nz.variable.map(S.label), sig=nz.p.map(sig))
        add(table(nz, {"label": "Indice", "group": "Groupe", "n": "n", "M": "M", "t": "t", "df": "ddl",
                       "p": "p", "sig": "", "d": "d de Cohen"}, {"p": pfmt, "df": lambda x: f"{x:.0f}",
                                                                  "n": lambda x: f"{x:.0f}"}))

    # ---------------------------------------------------------------- courbes d'apprentissage
    sec = 4
    if not curves.empty:
        add(f"<h2>{sec}. Courbes d'apprentissage</h2>")
        add(f"<figure>{learning_figure(curves, order, colors)}<figcaption>Figure {sec}. Erreurs par essai "
            "(moyenne et IC 95 %).</figcaption></figure>")
        for task, m in R["mixed"].items():
            name = "Hebb-Williams" if task == "maze" else "Apprentissage inversé"
            if not m or "error" in m:
                add(f"<p>{name} : modèle mixte non estimable ({html.escape(str(m.get('error', '')))}).</p>")
                continue
            add(f"<h3>{name} — modèle linéaire mixte</h3>")
            add(f"<p>Modèle <code>{html.escape(m['formula'])}</code>, effets aléatoires "
                f"<code>{html.escape(m['random'])}</code> par sujet ({m['n_subjects']} sujets, "
                f"{m['n_obs']} observations). Interaction groupe × essai (test de Wald) : "
                f"<b>{html.escape(m['apa_interaction'])}</b>. "
                + ("Les groupes apprennent à des vitesses différentes." if m["wald_p"] < S.ALPHA
                   else "Rien n'indique que les groupes apprennent à des vitesses différentes.") + "</p>")
            add("<details><summary>Coefficients</summary>")
            add(table(m["coefs"], {"term": "Terme", "coef": "b", "se": "ES", "z": "z", "p": "p"}, {"p": pfmt}))
            add("</details>")
        sec += 1

    # ---------------------------------------------------------------- analyses propres au plan
    key_vars = [v for v in ("brain_n_synapses", "brain_coverage_objects", "brain_pattern_separation",
                            "maze_errors") if v in df]
    if design == "dose" and "dose" in R:
        add(f"<h2>{sec}. Relation dose-réponse</h2>")
        add(f"<figure>{dose_figure(df, key_vars)}<figcaption>Figure {sec}. Moyenne et IC 95 % par niveau "
            "d'enrichissement.</figcaption></figure>")
        d = R["dose"].assign(sig=R["dose"].p_holm.map(sig))
        add(table(d, {"label": "Variable", "apa": "Régression linéaire ; Spearman", "p_holm": "p (Holm)",
                      "sig": "", "quad_p": "p terme quadratique"}, {"p_holm": pfmt, "quad_p": pfmt}))
        add('<p class="note">Un terme quadratique significatif indique une relation non linéaire '
            "(par ex. un plafond : les premiers niveaux d'enrichissement ont plus d'effet que les derniers).</p>")
        sec += 1
    if design == "timing" and "timing" in R:
        add(f"<h2>{sec}. Période critique (plan 2 × 2)</h2>")
        add(f"<figure>{timing_figure(df, key_vars)}<figcaption>Figure {sec}. Des droites non parallèles "
            "signalent une interaction précoce × tardif.</figcaption></figure>")
        t = R["timing"].copy()
        add(table(t, {"label": "Variable", "apa_early": "Effet précoce", "apa_late": "Effet tardif",
                      "apa_inter": "Interaction"}))
        add("<h3>Contraste clé : enrichi seulement tôt (EC→IC) vs seulement tard (IC→EC)</h3>")
        add(table(t, {"label": "Variable", "t_early_vs_late": "t de Welch", "p_early_vs_late": "p",
                      "g_early_vs_late": "g (tôt – tard)"}, {"p_early_vs_late": pfmt}))
        add('<p class="note">Avec une plasticité qui décroît avec l\'âge, l\'hypothèse de période '
            "critique prédit un effet précoce plus grand que l'effet tardif (η²p précoce > η²p tardif, "
            "g positif sur les variables où l'enrichissement augmente le score).</p>")
        sec += 1
    if design == "ablation" and "ablation" in R:
        add(f"<h2>{sec}. Contribution des composantes de l'enrichissement</h2>")
        add(f"<figure>{ablation_figure(R['ablation'], key_vars)}<figcaption>Figure {sec}. 0 % : retirer la "
            "composante ne change rien ; 100 % : le groupe tombe au niveau IC.</figcaption></figure>")
        a = R["ablation"].assign(sig=R["ablation"].p.map(sig))
        add("<details><summary>Tableau complet des tests de Dunnett (chaque groupe vs EC)</summary>")
        add(table(a[a.variable.isin(anat + behav)], {
            "label": "Variable", "group": "Groupe", "M": "M", "M_control": "M (EC)", "diff": "Diff.",
            "p": "p (Dunnett)", "sig": "", "g": "g", "pct_effect_lost": "% d'effet perdu"},
            {"p": pfmt, "pct_effect_lost": lambda x: S.fmt_num(x, 0)}))
        add("</details>")
        sec += 1

    # ---------------------------------------------------------------- développement
    if not dev.empty:
        add(f"<h2>{sec}. Trajectoires développementales</h2>")
        add(f"<figure>{development_figure(dev, order, colors)}<figcaption>Figure {sec}. Évolution des "
            "mesures pendant l'élevage (moyenne et IC 95 %)"
            + (" ; trait vertical : changement d'environnement" if "phase" in dev and dev.phase.max() > 0 else "")
            + ".</figcaption></figure>")
        sec += 1

    # ---------------------------------------------------------------- le cerveau pendant les tests
    st_path = folder / "test_development.csv"
    frozen = bool(meta.get("tests", {}).get("freeze_structure"))
    if st_path.exists() and "test_stages" in R:
        stages = pd.read_csv(st_path)
        add(f"<h2>{sec}. Le cerveau pendant la batterie de tests</h2>")
        if frozen:
            add('<p class="note">Dans ce plan, l\'architecture du cerveau (synapses, neurones) est figée '
                "pendant les tests : l'anatomie doit rester constante (contrôle de la manipulation). "
                "L'apprentissage des poids synaptiques continue.</p>")
        else:
            add('<p class="note">La plasticité structurelle reste active pendant les tests : la batterie '
                "dure plusieurs milliers d'étapes et continue de modifier le cerveau. Une baisse de η² au "
                "fil des tests signifie que les groupes se ressemblent de plus en plus.</p>")
        add(f"<figure>{test_stage_figure(stages, order, colors)}<figcaption>Figure {sec}. Mesures avant la "
            "batterie de tests puis après chaque test (moyenne et IC 95 %).</figcaption></figure>")
        ts = R["test_stages"]
        ts = ts[ts.variable.isin(list(STAGE_FIG_VARS) + ["n_neurons", "pattern_separation"])]
        wide = ts.pivot_table(index="label", columns="stage", values="eta2", sort=False)
        names = dict(zip(ts.stage, ts.stage_label))
        wide.columns = ["avant les tests" if c == 0 else f"après : {names[c]}" for c in wide.columns]
        wide = wide.reset_index().rename(columns={"label": "Mesure (η² du groupe)"})
        add(table(wide, {c: c for c in wide.columns}, {c: (lambda x: S.fmt_num(x, 2, lead_zero=False))
                                                       for c in wide.columns[1:]}))
        sec += 1

    # ---------------------------------------------------------------- contrôle vs protocole principal
    vs = R.get("vs_main")
    if vs is not None and not vs.empty:
        add(f"<h2>{sec}. Comparaison avec le protocole principal (analyse exploratoire)</h2>")
        same = R.get("same_rearing")
        add('<p class="note">Mêmes sujets que l\'expérience principale (même graine) : seul change le '
            "protocole de test. Élevage identique dans les deux fichiers : "
            f"<b>{'oui' if same else 'NON — comparaison non valide'}</b>. Effet du protocole : test t "
            "apparié sur la différence (architecture figée − plastique). Interaction groupe × protocole : "
            "comparaison des groupes sur cette différence. Analyse décidée après l'expérience finale : "
            "à présenter comme exploratoire.</p>")
        vt = vs.assign(sig=vs.inter_p_holm.map(sig))
        add(table(vt, {"label": "Variable", "eta2_base": "η² groupe (plastique)",
                       "eta2_other": "η² groupe (figé)", "mean_diff": "Diff. moyenne (figé − plastique)",
                       "apa_protocol": "Effet du protocole", "inter_apa": "Interaction groupe × protocole",
                       "inter_p_holm": "p interaction (Holm)", "sig": ""},
                  {"eta2_base": lambda x: S.fmt_num(x, 2, lead_zero=False),
                   "eta2_other": lambda x: S.fmt_num(x, 2, lead_zero=False),
                   "mean_diff": lambda x: S.fmt_num(x, 3), "inter_p_holm": pfmt}))
        sec += 1

    add(f"<h2>{sec}. Précautions d'interprétation</h2>")
    add("<ul><li>Il s'agit d'une simulation : les « sujets » sont des réseaux de neurones artificiels "
        "et les résultats ne se généralisent pas directement à l'animal ou à l'humain.</li>"
        "<li>La règle de plasticité « use it or lose it » (élagage des synapses dont l'entrée est "
        "silencieuse) rend une partie de l'effet anatomique attendue par construction. Le transfert "
        "vers le comportement, lui, n'est pas imposé par cette règle.</li>"
        + ("<li>L'architecture est figée pendant les tests dans ce plan de contrôle, ajouté après "
           "l'expérience finale : ses résultats sont exploratoires.</li>" if frozen else
           "<li>La plasticité reste active pendant les tests, qui durent plusieurs milliers d'étapes : "
           "les tests passés en dernier (labyrinthe, inversion) évaluent des cerveaux déjà modifiés par "
           "les tests précédents.</li>")
        + "<li>Les paramètres du modèle ont été calibrés lors d'une phase exploratoire (pilotes, graine "
        "2026) puis figés ; l'expérience rapportée ici utilise une graine différente.</li>"
        "<li>Avec un grand nombre de variables, des résultats isolés peuvent être des faux positifs : "
        "se fier aux p corrigés (Holm) et aux tailles d'effet.</li></ul>")

    doc = (f'<!doctype html><html lang="fr"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width, initial-scale=1">'
           f"<title>Rapport EnrichSim — {html.escape(design)}</title><style>{CSS}</style></head>"
           f"<body><main>{''.join(parts)}</main></body></html>")
    path = folder / "rapport.html"
    path.write_text(doc, encoding="utf-8")
    return path
