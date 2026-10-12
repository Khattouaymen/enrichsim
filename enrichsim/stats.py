"""Analyses statistiques des données simulées.

Le plan d'analyse est fixé AVANT l'exécution finale (graine 2027) :

* statistiques descriptives : M, ET, IC 95 %, médiane ;
* conditions d'application : Shapiro-Wilk (normalité, par groupe) et Levene (homogénéité) ;
* test global choisi selon les conditions :
    - normalité ET homogénéité  -> ANOVA de Fisher,      post-hoc de Tukey HSD ;
    - normalité sans homogénéité -> ANOVA de Welch,      post-hoc de Games-Howell ;
    - normalité rejetée          -> Kruskal-Wallis,      post-hoc Mann-Whitney + Holm ;
* tailles d'effet : η², ω² (ou ε² pour Kruskal-Wallis), g de Hedges pour chaque paire ;
* correction de Holm des p globaux au sein de chaque famille de variables
  (anatomie / comportement) pour limiter l'inflation du risque α ;
* courbes d'apprentissage : modèle linéaire mixte groupe x essai (sujet en effet aléatoire) ;
* dose-réponse : régression linéaire (et quadratique) + corrélation de Spearman ;
* période critique : ANOVA 2 x 2 (précoce x tardif) de type II, η² partiel ;
* ablation : test de Dunnett, chaque groupe comparé à EC.

Remarque sur le plan apparié : le sujet n°i de chaque groupe démarre avec le même cerveau
initial. L'analyse principale traite pourtant les groupes comme indépendants (comme dans les
études animales) ; c'est conservateur. Une analyse complémentaire (Friedman) exploite
l'appariement.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05

# variable -> (libellé français, famille)
VARIABLES: dict[str, tuple[str, str]] = {
    # anatomie / représentations
    "brain_n_neurons": ("Nombre de neurones", "anatomie"),
    "brain_n_synapses": ("Nombre de synapses", "anatomie"),
    "brain_fan_in_mean": ("Complexité dendritique (synapses / neurone)", "anatomie"),
    "brain_synaptic_strength": ("Force synaptique moyenne", "anatomie"),
    "brain_coverage_objects": ("Territoire cortical « objets »", "anatomie"),
    "brain_dimensionality": ("Dimensionnalité des représentations", "anatomie"),
    "brain_pattern_separation": ("Séparation de patterns", "anatomie"),
    "brain_object_sensitivity": ("Sensibilité corticale aux objets", "anatomie"),
    "nor_neural_di": ("Indice de discrimination neuronal (NOR)", "anatomie"),
    # comportement
    "of_exploration": ("Champ ouvert : exploration", "comportement"),
    "of_center_time": ("Champ ouvert : temps au centre", "comportement"),
    "nor_exploration": ("NOR : temps d'exploration des objets", "comportement"),
    "nor_di": ("NOR : indice de discrimination comportemental", "comportement"),
    "maze_errors": ("Hebb-Williams : erreurs par essai", "comportement"),
    "maze_trials_to_criterion": ("Hebb-Williams : essais jusqu'au critère", "comportement"),
    "maze_latency": ("Hebb-Williams : latence (pas)", "comportement"),
    "reversal_errors": ("Inversion : erreurs par essai", "comportement"),
    "reversal_perseveration": ("Inversion : persévération", "comportement"),
}


def label(var: str) -> str:
    return VARIABLES.get(var, (var, ""))[0]


def family_vars(df: pd.DataFrame, family: str) -> list[str]:
    return [v for v, (_, f) in VARIABLES.items() if f == family and v in df.columns]


# =========================================================================== mise en forme APA

def fmt_p(p: float) -> str:
    """p au format APA : sans zéro initial, « p < .001 » sous le seuil."""
    if p is None or not np.isfinite(p):
        return "p = n.d."
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".", 1)


def fmt_num(x: float, d: int = 2, lead_zero: bool = True) -> str:
    if x is None or not np.isfinite(x):
        return "n.d."
    s = f"{x:.{d}f}"
    if not lead_zero:
        s = s.replace("0.", ".", 1)
    return s


def fmt_df(d: float) -> str:
    return f"{d:.0f}" if abs(d - round(d)) < 1e-9 else f"{d:.2f}"


# =========================================================================== briques de base

def _clean(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[np.isfinite(a)]


def describe(df: pd.DataFrame, var: str, group: str = "group",
             order: list[str] | None = None) -> pd.DataFrame:
    order = order or list(dict.fromkeys(df[group]))
    rows = []
    for g in order:
        x = _clean(df.loc[df[group] == g, var])
        n = len(x)
        m = x.mean() if n else np.nan
        sd = x.std(ddof=1) if n > 1 else np.nan
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
        rows.append({"variable": var, "group": g, "n": n, "M": m, "SD": sd,
                     "CI_low": m - half, "CI_high": m + half,
                     "median": np.median(x) if n else np.nan})
    return pd.DataFrame(rows)


def hedges_g(a, b) -> float:
    """g de Hedges (d de Cohen corrigé du biais des petits échantillons). Signe : a - b."""
    a, b = _clean(a), _clean(b)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if sp == 0:
        return np.nan
    d = (a.mean() - b.mean()) / sp
    return float(d * (1 - 3 / (4 * (n1 + n2) - 9)))


def holm(pvals) -> np.ndarray:
    """Correction de Holm-Bonferroni (les NaN sont ignorés)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        out[i] = min(1.0, running)
    return out


def check_assumptions(samples: list[np.ndarray]) -> dict:
    shapiro = []
    for x in samples:
        if len(x) >= 3 and np.ptp(x) > 0:
            shapiro.append(float(stats.shapiro(x).pvalue))
        else:
            shapiro.append(np.nan)
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            lev = float(stats.levene(*samples, center="median").pvalue)
    except Exception:
        lev = np.nan
    finite = [p for p in shapiro if np.isfinite(p)]
    normal = bool(finite) and min(finite) > ALPHA
    homogeneous = (not np.isfinite(lev)) or lev > ALPHA
    return {"shapiro_min_p": min(finite) if finite else np.nan, "shapiro": shapiro,
            "levene_p": lev, "normal": normal, "homogeneous": homogeneous}


def welch_anova(samples: list[np.ndarray]) -> tuple[float, float, float, float]:
    """ANOVA de Welch (1951) : F, ddl1, ddl2, p."""
    k = len(samples)
    n = np.array([len(x) for x in samples], float)
    m = np.array([x.mean() for x in samples])
    v = np.array([x.var(ddof=1) for x in samples])
    if np.any(v == 0):
        v = np.where(v == 0, 1e-12, v)
    w = n / v
    mw = (w * m).sum() / w.sum()
    a = (w * (m - mw) ** 2).sum() / (k - 1)
    tmp = (((1 - w / w.sum()) ** 2) / (n - 1)).sum()
    b = 1 + 2 * (k - 2) / (k ** 2 - 1) * tmp
    F = a / b
    df2 = (k ** 2 - 1) / (3 * tmp)
    return float(F), float(k - 1), float(df2), float(stats.f.sf(F, k - 1, df2))


def anova_effect_sizes(samples: list[np.ndarray]) -> tuple[float, float]:
    """η² et ω² d'une ANOVA à un facteur."""
    allx = np.concatenate(samples)
    gm = allx.mean()
    ss_b = sum(len(x) * (x.mean() - gm) ** 2 for x in samples)
    ss_t = ((allx - gm) ** 2).sum()
    k, N = len(samples), len(allx)
    if ss_t == 0:
        return np.nan, np.nan
    ms_w = (ss_t - ss_b) / (N - k)
    eta2 = ss_b / ss_t
    omega2 = (ss_b - (k - 1) * ms_w) / (ss_t + ms_w)
    return float(eta2), float(max(omega2, 0.0))


def games_howell(samples: list[np.ndarray], names: list[str]) -> list[dict]:
    out = []
    k = len(samples)
    for i in range(k):
        for j in range(i + 1, k):
            a, b = samples[i], samples[j]
            va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
            se = np.sqrt((va + vb) / 2)
            diff = a.mean() - b.mean()
            if se == 0:
                p = np.nan if diff == 0 else 0.0
            else:
                dfw = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
                q = abs(diff) / se
                p = float(stats.studentized_range.sf(q, k, dfw))
            out.append({"A": names[i], "B": names[j], "diff": float(diff), "p": p})
    return out


def tukey(samples: list[np.ndarray], names: list[str]) -> list[dict]:
    res = stats.tukey_hsd(*samples)
    out = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            out.append({"A": names[i], "B": names[j],
                        "diff": float(samples[i].mean() - samples[j].mean()),
                        "p": float(res.pvalue[i, j])})
    return out


def mann_whitney_holm(samples: list[np.ndarray], names: list[str]) -> list[dict]:
    out = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            a, b = samples[i], samples[j]
            try:
                p = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
            except ValueError:
                p = np.nan
            out.append({"A": names[i], "B": names[j],
                        "diff": float(np.median(a) - np.median(b)), "p": p})
    for r, ph in zip(out, holm([r["p"] for r in out])):
        r["p"] = float(ph)
    return out


# =========================================================================== comparaison de k groupes

@dataclass
class OmnibusResult:
    variable: str
    test: str
    stat: float
    df1: float
    df2: float
    p: float
    eta2: float
    omega2: float
    shapiro_min_p: float
    levene_p: float
    posthoc_method: str
    posthoc: list[dict]
    friedman_p: float = np.nan

    def apa(self) -> str:
        if self.test == "Kruskal-Wallis":
            return (f"H({fmt_df(self.df1)}) = {fmt_num(self.stat)}, {fmt_p(self.p)}, "
                    f"ε² = {fmt_num(self.eta2, lead_zero=False)}")
        prefix = "F" if self.test == "ANOVA" else "F de Welch"
        return (f"{prefix}({fmt_df(self.df1)}, {fmt_df(self.df2)}) = {fmt_num(self.stat)}, "
                f"{fmt_p(self.p)}, η² = {fmt_num(self.eta2, lead_zero=False)}, "
                f"ω² = {fmt_num(self.omega2, lead_zero=False)}")

    def row(self) -> dict:
        return {"variable": self.variable, "label": label(self.variable), "test": self.test,
                "stat": self.stat, "df1": self.df1, "df2": self.df2, "p": self.p,
                "eta2": self.eta2, "omega2": self.omega2,
                "shapiro_min_p": self.shapiro_min_p, "levene_p": self.levene_p,
                "posthoc": self.posthoc_method, "friedman_p": self.friedman_p,
                "apa": self.apa()}


def compare_groups(df: pd.DataFrame, var: str, group: str = "group",
                   order: list[str] | None = None, paired_by: str | None = "subject") -> OmnibusResult | None:
    order = order or list(dict.fromkeys(df[group]))
    samples = [_clean(df.loc[df[group] == g, var]) for g in order]
    if any(len(x) < 2 for x in samples) or np.ptp(np.concatenate(samples)) == 0:
        return None
    a = check_assumptions(samples)
    eta2, omega2 = anova_effect_sizes(samples)
    if a["normal"] and a["homogeneous"]:
        F, p = stats.f_oneway(*samples)
        N = sum(len(x) for x in samples)
        res = OmnibusResult(var, "ANOVA", float(F), len(samples) - 1, N - len(samples), float(p),
                            eta2, omega2, a["shapiro_min_p"], a["levene_p"], "Tukey HSD",
                            tukey(samples, order))
    elif a["normal"]:
        F, d1, d2, p = welch_anova(samples)
        res = OmnibusResult(var, "Welch", F, d1, d2, p, eta2, omega2, a["shapiro_min_p"],
                            a["levene_p"], "Games-Howell", games_howell(samples, order))
    else:
        H, p = stats.kruskal(*samples)
        N = sum(len(x) for x in samples)
        eps2 = float(H / (N - 1))  # ε² (Tomczak & Tomczak, 2014)
        res = OmnibusResult(var, "Kruskal-Wallis", float(H), len(samples) - 1, np.nan, float(p),
                            eps2, np.nan, a["shapiro_min_p"], a["levene_p"],
                            "Mann-Whitney (Holm)", mann_whitney_holm(samples, order))
    for r in res.posthoc:
        r["g"] = hedges_g(df.loc[df[group] == r["A"], var], df.loc[df[group] == r["B"], var])
    # analyse complémentaire exploitant l'appariement (même cerveau initial)
    if paired_by and paired_by in df.columns:
        wide = df.pivot_table(index=paired_by, columns=group, values=var)[order].dropna()
        if len(wide) >= 3 and len(order) >= 3:
            try:
                res.friedman_p = float(stats.friedmanchisquare(*[wide[g] for g in order]).pvalue)
            except ValueError:
                pass
    return res


def compare_many(df: pd.DataFrame, variables: list[str], group: str = "group",
                 order: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tests globaux + post-hoc pour une liste de variables ; p de Holm par famille."""
    omni, post = [], []
    for v in variables:
        r = compare_groups(df, v, group, order)
        if r is None:
            continue
        omni.append(r.row())
        for ph in r.posthoc:
            post.append({"variable": v, "label": label(v), "method": r.posthoc_method, **ph})
    omni_df = pd.DataFrame(omni)
    if not omni_df.empty:
        omni_df["family"] = omni_df["variable"].map(lambda v: VARIABLES.get(v, ("", "autre"))[1])
        omni_df["p_holm"] = np.nan
        for fam, idx in omni_df.groupby("family").groups.items():
            omni_df.loc[idx, "p_holm"] = holm(omni_df.loc[idx, "p"])
    return omni_df, pd.DataFrame(post)


def one_sample_vs_zero(df: pd.DataFrame, var: str, group: str = "group",
                       order: list[str] | None = None) -> pd.DataFrame:
    """Le groupe préfère-t-il l'objet nouveau ? (indice de discrimination > 0)"""
    order = order or list(dict.fromkeys(df[group]))
    rows = []
    for g in order:
        x = _clean(df.loc[df[group] == g, var])
        if len(x) < 2 or x.std() == 0:
            continue
        t, p = stats.ttest_1samp(x, 0.0)
        rows.append({"variable": var, "group": g, "n": len(x), "M": x.mean(), "t": float(t),
                     "df": len(x) - 1, "p": float(p), "d": float(x.mean() / x.std(ddof=1))})
    return pd.DataFrame(rows)


# =========================================================================== courbes d'apprentissage

def learning_mixed_model(curves: pd.DataFrame, task: str = "maze", dv: str = "errors",
                         order: list[str] | None = None) -> dict:
    """Modèle linéaire mixte : dv ~ groupe * essai (+ problème), sujet en effet aléatoire.

    L'essai est centré. La pente aléatoire par sujet est tentée puis abandonnée si le
    modèle ne converge pas. Le test de Wald conjoint des termes d'interaction groupe x essai
    indique si les groupes apprennent à des vitesses différentes.
    """
    import statsmodels.formula.api as smf

    d = curves[curves["task"] == task].copy()
    if d.empty or d["group"].nunique() < 2:
        return {}
    order = order or list(dict.fromkeys(d["group"]))
    d["group"] = pd.Categorical(d["group"], categories=order)
    d["trial_c"] = d["trial"] - d["trial"].mean()
    d["uid"] = d["group"].astype(str) + "_" + d["subject"].astype(str)
    has_prob = d["problem"].nunique() > 1
    base = f"{dv} ~ C(group) * trial_c" + (" + C(problem)" if has_prob else "")

    def fit(formula, re, reml, method):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return smf.mixedlm(formula, d, groups=d["uid"], re_formula=re).fit(reml=reml, method=method)

    full, re_used = None, None
    for re, method in (("~trial_c", "lbfgs"), ("~1", "lbfgs"), ("~1", "powell")):
        try:
            m = fit(base, re, True, method)
            if m.converged and np.all(np.isfinite(m.bse_fe)):
                full, re_used = m, re
                break
        except Exception:
            continue
    if full is None:
        return {"error": "le modèle mixte n'a pas convergé"}
    # test de Wald conjoint des termes d'interaction groupe x essai
    terms = [t for t in full.fe_params.index if ":trial_c" in t]
    w_df = len(terms)
    try:
        bvec = full.fe_params[terms].to_numpy()
        V = full.cov_params().loc[terms, terms].to_numpy()
        w = float(bvec @ np.linalg.solve(V, bvec))
        w_p = float(stats.chi2.sf(w, w_df))
    except Exception:
        w, w_p = np.nan, np.nan
    coefs = pd.DataFrame({"coef": full.fe_params, "se": full.bse_fe,
                          "z": full.fe_params / full.bse_fe,
                          "p": full.pvalues[full.fe_params.index]})
    return {
        "task": task, "dv": dv, "formula": base, "random": re_used,
        "converged": bool(full.converged), "n_obs": int(full.nobs),
        "n_subjects": int(d["uid"].nunique()), "coefs": coefs.reset_index(names="term"),
        "wald_interaction": w, "wald_df": w_df, "wald_p": w_p,
        "apa_interaction": f"χ²({w_df}) = {fmt_num(w)}, {fmt_p(w_p)}",
    }


# =========================================================================== analyses par plan

def analyze_dose(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    import statsmodels.formula.api as smf

    rows = []
    for v in variables:
        d = df[["level", v]].dropna()
        if len(d) < 5 or d[v].std() == 0:
            continue
        lin = smf.ols(f"{v} ~ level", d).fit()
        quad = smf.ols(f"{v} ~ level + I(level ** 2)", d).fit()
        rho, p_rho = stats.spearmanr(d["level"], d[v])
        rows.append({
            "variable": v, "label": label(v), "family": VARIABLES.get(v, ("", "autre"))[1],
            "slope": lin.params["level"], "slope_se": lin.bse["level"],
            "t": lin.tvalues["level"], "df": lin.df_resid, "p": lin.pvalues["level"],
            "r2": lin.rsquared, "quad_coef": quad.params["I(level ** 2)"],
            "quad_p": quad.pvalues["I(level ** 2)"], "rho": rho, "rho_p": p_rho,
            "apa": (f"b = {fmt_num(lin.params['level'], 3)}, t({fmt_df(lin.df_resid)}) = "
                    f"{fmt_num(lin.tvalues['level'])}, {fmt_p(lin.pvalues['level'])}, "
                    f"R² = {fmt_num(lin.rsquared, lead_zero=False)} ; ρ = "
                    f"{fmt_num(rho, lead_zero=False)}"),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        for fam, idx in out.groupby("family").groups.items():
            out.loc[idx, "p_holm"] = holm(out.loc[idx, "p"])
    return out


def analyze_timing(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """ANOVA 2 x 2 (précoce x tardif). Contraste clé de la période critique :
    EC→IC (enrichi seulement tôt) vs IC→EC (enrichi seulement tard)."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    rows = []
    for v in variables:
        d = df[["early", "late", "group", v]].dropna()
        if d[v].std() == 0 or d.groupby(["early", "late"]).size().min() < 2:
            continue
        model = smf.ols(f"{v} ~ C(early) * C(late)", d).fit()
        tab = sm.stats.anova_lm(model, typ=2)
        ss_res = tab.loc["Residual", "sum_sq"]
        df_res = tab.loc["Residual", "df"]
        row = {"variable": v, "label": label(v), "family": VARIABLES.get(v, ("", "autre"))[1]}
        for term, key in (("C(early)", "early"), ("C(late)", "late"), ("C(early):C(late)", "inter")):
            ss = tab.loc[term, "sum_sq"]
            row[f"F_{key}"] = tab.loc[term, "F"]
            row[f"p_{key}"] = tab.loc[term, "PR(>F)"]
            row[f"eta2p_{key}"] = ss / (ss + ss_res)
            row[f"apa_{key}"] = (f"F(1, {fmt_df(df_res)}) = {fmt_num(tab.loc[term, 'F'])}, "
                                 f"{fmt_p(tab.loc[term, 'PR(>F)'])}, η²p = "
                                 f"{fmt_num(ss / (ss + ss_res), lead_zero=False)}")
        a = d.loc[(d.early == 1) & (d.late == 0), v]
        b = d.loc[(d.early == 0) & (d.late == 1), v]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        row.update(t_early_vs_late=float(t), p_early_vs_late=float(p), g_early_vs_late=hedges_g(a, b))
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_ablation(df: pd.DataFrame, variables: list[str], control: str = "EC",
                     floor: str = "IC") -> pd.DataFrame:
    """Dunnett : chaque condition comparée à EC. `pct_effect_lost` = part de l'écart EC - IC
    perdue quand on retire la composante (0 % : sans importance ; 100 % : comme IC).
    Non calculé (NaN) quand EC ne diffère pas significativement de IC sur la variable."""
    groups = [g for g in dict.fromkeys(df["group"]) if g != control]
    rows = []
    for v in variables:
        ctrl = _clean(df.loc[df.group == control, v])
        others = [_clean(df.loc[df.group == g, v]) for g in groups]
        if len(ctrl) < 2 or any(len(o) < 2 for o in others):
            continue
        if np.ptp(np.concatenate([ctrl, *others])) == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = stats.dunnett(*others, control=ctrl)
        full = ctrl.mean() - _clean(df.loc[df.group == floor, v]).mean()
        for g, o, p, st in zip(groups, others, res.pvalue, res.statistic):
            rows.append({
                "variable": v, "label": label(v), "group": g, "M": o.mean(), "M_control": ctrl.mean(),
                "diff": o.mean() - ctrl.mean(), "stat": float(st), "p": float(p),
                "g": hedges_g(o, ctrl),
                "pct_effect_lost": float(100 * (ctrl.mean() - o.mean()) / full) if full else np.nan,
            })
        # le pourcentage n'a de sens que si EC diffère réellement de IC sur cette variable
        floor_rows = [r for r in rows if r["variable"] == v and r["group"] == floor]
        if not floor_rows or floor_rows[0]["p"] >= ALPHA:
            for r in rows:
                if r["variable"] == v:
                    r["pct_effect_lost"] = np.nan
    return pd.DataFrame(rows)


# =========================================================================== point d'entrée

def _order(df: pd.DataFrame, meta: dict) -> list[str]:
    names = [g["name"] for g in meta.get("groups", [])]
    present = list(dict.fromkeys(df["group"]))
    return [n for n in names if n in present] or present


def analyze(folder: str | Path, save: bool = True) -> dict:
    """Analyse complète d'un dossier de résultats. Écrit les tableaux dans `folder/stats/`."""
    folder = Path(folder)
    df = pd.read_csv(folder / "subjects.csv")
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    curves_path = folder / "learning_curves.csv"
    curves = pd.read_csv(curves_path) if curves_path.exists() else pd.DataFrame()
    design = meta["design"]
    order = _order(df, meta)
    anat, behav = family_vars(df, "anatomie"), family_vars(df, "comportement")
    allv = anat + behav

    out: dict = {"design": design, "order": order, "meta": meta}
    out["descriptives"] = pd.concat([describe(df, v, order=order) for v in allv], ignore_index=True)
    out["omnibus"], out["posthoc"] = compare_many(df, allv, order=order)
    out["nor_vs_zero"] = pd.concat([one_sample_vs_zero(df, v, order=order)
                                    for v in ("nor_di", "nor_neural_di") if v in df], ignore_index=True)
    out["mixed"] = {}
    if not curves.empty:
        for task in ("maze", "reversal"):
            try:
                out["mixed"][task] = learning_mixed_model(curves, task, "errors", order)
            except Exception as e:  # le modèle mixte peut échouer sur de très petits jeux de données
                out["mixed"][task] = {"error": str(e)}
    if design == "dose":
        out["dose"] = analyze_dose(df, allv)
    if design == "timing":
        out["timing"] = analyze_timing(df, allv)
    if design == "ablation":
        out["ablation"] = analyze_ablation(df, allv)

    if save:
        sd = folder / "stats"
        sd.mkdir(exist_ok=True)
        for k, v in out.items():
            if isinstance(v, pd.DataFrame) and not v.empty:
                v.to_csv(sd / f"{k}.csv", index=False)
        for task, m in out["mixed"].items():
            if "coefs" in m:
                m["coefs"].to_csv(sd / f"mixed_{task}.csv", index=False)
    return out
