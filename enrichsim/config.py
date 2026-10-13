"""Configuration du modèle : environnements, cerveau et protocoles expérimentaux.

Les trois conditions reprennent le protocole historique de Rosenzweig, Bennett & Diamond
(1972) :

* IC (Impoverished Condition) : isolement, petite cage vide, aucune nouveauté ;
* SC (Standard Condition)     : cage standard, quelques congénères, pas d'objets ;
* EC (Enriched Condition)     : grand espace, congénères, objets variés, réaménagement
  régulier (nouveauté).

Chaque paramètre de l'environnement correspond à une *composante* de l'enrichissement,
ce qui permet de les manipuler séparément (expérience d'ablation) ou de façon continue
(expérience dose-réponse).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class EnvConfig:
    """Paramètres d'une cage d'élevage (phase de développement)."""

    name: str = "SC"
    size: int = 10                 # côté de la cage (cases) -> composante « espace »
    n_objects: int = 0             # objets présents simultanément -> « complexité »
    object_pool: int = 0           # nombre de types d'objets disponibles -> « diversité »
    novelty_period: int = 0        # réaménagement toutes les N étapes (0 = jamais) -> « nouveauté »
    n_conspecifics: int = 0        # congénères -> composante « sociale »
    inner_walls: int = 0           # tunnels / cloisons -> complexité spatiale
    food_density: float = 0.04     # nourriture par case (identique entre conditions)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- Conditions de référence -------------------------------------------------------------
IC = EnvConfig(name="IC", size=6, n_objects=0, object_pool=0, novelty_period=0,
               n_conspecifics=0, inner_walls=0)
SC = EnvConfig(name="SC", size=10, n_objects=0, object_pool=0, novelty_period=0,
               n_conspecifics=2, inner_walls=0)
EC = EnvConfig(name="EC", size=16, n_objects=10, object_pool=40, novelty_period=400,
               n_conspecifics=4, inner_walls=6)

CONDITIONS: dict[str, EnvConfig] = {"IC": IC, "SC": SC, "EC": EC}


def interpolate_enrichment(level: float, name: str | None = None) -> EnvConfig:
    """Environnement d'intensité continue entre IC (0.0) et EC (1.0) — dose-réponse."""
    level = float(min(max(level, 0.0), 1.0))

    def lerp(a: float, b: float) -> float:
        return a + (b - a) * level

    return EnvConfig(
        name=name or f"E{level:.2f}",
        size=int(round(lerp(IC.size, EC.size))),
        n_objects=int(round(lerp(0, EC.n_objects))),
        object_pool=int(round(lerp(0, EC.object_pool))),
        # la nouveauté apparaît dès qu'il y a des objets ; plus le niveau est haut, plus elle est fréquente
        novelty_period=0 if level < 0.05 else int(round(lerp(2000, EC.novelty_period))),
        n_conspecifics=int(round(lerp(0, EC.n_conspecifics))),
        inner_walls=int(round(lerp(0, EC.inner_walls))),
    )


def ablate(base: EnvConfig, component: str) -> EnvConfig:
    """Retire une seule composante de l'enrichissement (plan d'ablation)."""
    if component == "objects":
        return replace(base, name=f"{base.name}-objets", n_objects=0, object_pool=0, novelty_period=0)
    if component == "novelty":
        return replace(base, name=f"{base.name}-nouveauté", novelty_period=0)
    if component == "social":
        return replace(base, name=f"{base.name}-social", n_conspecifics=0)
    if component == "space":
        return replace(base, name=f"{base.name}-espace", size=IC.size + 2, inner_walls=0,
                       n_objects=min(base.n_objects, 3))
    raise ValueError(f"Composante inconnue : {component}")


@dataclass(frozen=True)
class BrainConfig:
    """Paramètres du cerveau artificiel (identiques pour tous les sujets)."""

    view_radius: int = 2           # champ perceptif : (2r+1)^2 cases
    object_features: int = 6       # dimension des traits sensoriels d'un objet
    place_cells: int = 4           # grille p x p de cellules de lieu (hippocampe)
    n_initial: int = 64            # neurones actifs à la naissance
    n_max: int = 112               # neurones possibles (réserve pour la neurogenèse)
    initial_density: float = 0.35  # probabilité de connexion initiale entrée -> neurone
    max_fan_in: int = 120          # capacité dendritique maximale d'un neurone

    # apprentissage fonctionnel (plasticité synaptique)
    lr_actor: float = 0.1          # pas normalisés par l'énergie de l'activité corticale
    lr_critic: float = 0.1
    lr_body: float = 0.02          # plasticité corticale guidée par la récompense
    lr_forward: float = 0.2        # règle delta normalisée (NLMS)
    gamma: float = 0.95
    trace_lambda: float = 0.8      # traces d'éligibilité (étiquetage synaptique)
    entropy_coef: float = 0.03
    motor_persistence: float = 1.0  # biais inné à continuer tout droit (identique pour tous)
    curiosity_coef: float = 0.1    # poids de la récompense intrinsèque (curiosité)
    memory_size: int = 400         # capacité de la mémoire de reconnaissance (états récents)
    object_memory_size: int = 60   # capacité de la mémoire des objets (voie du « quoi »)
    habituation_rate: float = 0.25  # renforcement d'un souvenir d'objet à chaque exposition
    investigation: float = 2.0     # prolongation de l'investigation d'un objet perçu comme nouveau
    weight_decay: float = 2e-5     # oubli passif des synapses non utilisées

    # réactivation hippocampique (replay)
    replay_passes: int = 3         # passes de replay après chaque essai de test
    sleep_replay: int = 300        # transitions rejouées à chaque « sommeil » pendant l'élevage
    replay_enabled: bool = False   # désactivé par défaut (voir expérience complémentaire)
    replay_actor: bool = True      # le replay met aussi à jour la politique

    # plasticité structurelle
    structural_period: int = 250   # fréquence des remaniements (étapes)
    coact_tau: float = 0.002       # constante de la moyenne mobile de co-activité
    prune_threshold: float = 0.02  # |w| sous lequel une synapse est « faible »
    prune_prob: float = 0.5        # élagage d'une synapse faible dont l'entrée est silencieuse
    prune_prob_used: float = 0.05  # élagage d'une synapse faible dont l'entrée est active
    pre_threshold: float = 0.003   # activité présynaptique moyenne minimale (« utilisée »)
    grow_rate: float = 0.02        # taux de synaptogenèse dépendante de l'activité
    neurogenesis_rate: float = 0.5  # probabilité de recruter un neurone si nouveauté élevée
    novelty_threshold: float = 1.15  # erreur de prédiction / référence déclenchant la neurogenèse
    silent_threshold: float = 0.01  # activité moyenne sous laquelle un neurone meurt
    target_rate: float = 0.15      # consigne d'activité moyenne (homéostasie)
    homeostasis: float = 0.05      # ajustement maximal des poids par remaniement
    min_neurons: int = 16

    # période critique : la plasticité décroît avec l'âge
    critical_period: bool = False
    plasticity_tau: float = 15000.0
    plasticity_floor: float = 0.25

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TestConfig:
    """Paramètres de la batterie de tests comportementaux."""

    open_field_steps: int = 300
    nor_sessions: int = 4
    nor_familiarization: int = 150
    nor_test: int = 200
    maze_trials: int = 8
    maze_max_steps: int = 200
    maze_problems: tuple[int, ...] = (0, 1, 2, 3)
    reversal_trials: int = 8
    # La plasticité reste active pendant les tests (protocole principal). True : l'architecture
    # du cerveau (synapses, neurones) est figée pendant les tests (expérience de contrôle).
    freeze_structure: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Phase:
    """Une phase d'élevage : un environnement pendant un certain nombre d'étapes."""

    env: EnvConfig
    steps: int


@dataclass
class Group:
    """Un groupe expérimental : une suite de phases d'élevage."""

    name: str
    phases: list[Phase]
    brain: BrainConfig = field(default_factory=BrainConfig)
    factors: dict[str, Any] = field(default_factory=dict)  # facteurs du plan (analyse factorielle)
