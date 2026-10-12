"""Cerveau artificiel à plasticité fonctionnelle ET structurelle.

Architecture (inspirée du couple cortex / ganglions de la base / hippocampe) :

    entrées sensorielles x ──(W1, masque M)──► couche corticale h (neurones ReLU)
                                                 ├─► politique π(a|h)   (acteur, striatum dorsal)
                                                 ├─► valeur V(h)         (critique, striatum ventral / dopamine)
                                                 └─► modèle prédictif  x̂(t+1) = Wf·[h, a]  (cerveau prédictif)

Plasticité fonctionnelle (à chaque étape) :
    * apprentissage par renforcement acteur-critique, l'erreur de prédiction de récompense δ
      jouant le rôle du signal dopaminergique ; traces d'éligibilité = étiquetage synaptique ;
    * apprentissage prédictif : le cortex apprend à prédire la prochaine sensation ;
    * mémoire de reconnaissance : les états corticaux récents sont mémorisés ; la
      nouveauté d'une situation est sa distance au souvenir le plus proche. Deux mémoires
      distinctes : la situation globale (« où », hippocampe) et l'identité des objets
      touchés (« quoi », cortex périrhinal ; Brown & Aggleton, 2001). La nouveauté
      contextuelle, rapportée à sa valeur habituelle, sert de récompense intrinsèque
      (curiosité) ; la nouveauté d'objet module le temps d'investigation. Un cortex qui
      ne code pas un attribut ne peut pas en percevoir la nouveauté.

Plasticité structurelle (toutes les `structural_period` étapes) :
    * synaptogenèse dépendante de la co-activité pré/post (« fire together, wire together ») ;
    * élagage des synapses faibles et inutilisées (« use it or lose it ») ;
    * neurogenèse déclenchée par la nouveauté (erreur de prédiction élevée, cf. gyrus denté) ;
    * mort des neurones silencieux (apoptose) ;
    * capacité dendritique maximale par neurone ;
    * plasticité homéostatique (« synaptic scaling », Turrigiano, 2008) qui maintient
      l'activité de chaque neurone près d'une consigne.

Les neurones ont un taux de décharge saturant : h = tanh(max(z, 0)).

Une période critique optionnelle fait décroître toute la plasticité avec l'âge.
"""
from __future__ import annotations

import numpy as np

from .config import BrainConfig
from .environment import N_ACTIONS

PE_SCALE = 0.01   # échelle de référence de l'erreur de prédiction sensorielle
F32 = np.float32  # les grandes matrices synaptiques sont en simple précision (vitesse)
COACT_EVERY = 4   # la co-activité est échantillonnée toutes les 4 étapes
LOGIT_MAX = 4.0   # borne des préférences d'action (évite un comportement figé)
NOV_SCALE = 0.15  # échelle de la nouveauté (distance euclidienne entre états corticaux)
OBJ_SCALE = 0.05  # échelle de la nouveauté d'objet (voie du « quoi »)


class PlasticBrain:
    def __init__(self, cfg: BrainConfig, n_inputs: int, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.D = n_inputs
        H = self.H = cfg.n_max
        A = N_ACTIONS

        self.alive = np.zeros(H, dtype=bool)
        self.alive[: cfg.n_initial] = True
        self.M = (rng.random((H, n_inputs)) < cfg.initial_density) & self.alive[:, None]
        std = 1.0 / np.sqrt(n_inputs * cfg.initial_density)
        self.W1 = (rng.normal(0.0, std, (H, n_inputs)) * self.M).astype(F32)
        self.b1 = np.zeros(H, dtype=F32)
        self.Wp = rng.normal(0.0, 0.01, (A, H))
        self.bp = np.zeros(A)
        self.wv = np.zeros(H)
        self.bv = 0.0
        self.Wf = np.zeros((n_inputs, H + A), dtype=F32)

        self._reset_traces()
        # moyennes mobiles d'activité (pour la plasticité structurelle)
        self.pre = np.zeros(n_inputs)
        self.post = np.zeros(H)
        self.coact = np.zeros((H, n_inputs), dtype=F32)
        self.pe_ema = PE_SCALE
        # mémoire de reconnaissance : tampon circulaire d'états corticaux récents
        self.memory = np.zeros((cfg.memory_size, H), dtype=F32)
        self.memory_n = 0
        self.memory_i = 0
        self.nov_ema = NOV_SCALE
        self.obj_memory = np.zeros((cfg.object_memory_size, H), dtype=F32)
        self.obj_strength = np.zeros(cfg.object_memory_size)  # force d'encodage (habituation)
        self.obj_memory_n = 0
        self.obj_memory_i = 0
        self.birth = np.zeros(H)  # âge de naissance de chaque neurone
        self.age = 0
        self.births = 0
        self.deaths = 0
        self.synapses_formed = 0
        self.synapses_pruned = 0

    # ------------------------------------------------------------------ utilitaires
    def _reset_traces(self) -> None:
        self.eW1 = np.zeros((self.H, self.D), dtype=F32)
        self.eWp = np.zeros_like(self.Wp)
        self.ebp = np.zeros(N_ACTIONS)
        self.ewv = np.zeros(self.H)
        self.ebv = 0.0

    def end_episode(self) -> None:
        self._reset_traces()

    def plasticity(self) -> float:
        """Facteur de plasticité selon l'âge (1 si pas de période critique)."""
        c = self.cfg
        if not c.critical_period:
            return 1.0
        return c.plasticity_floor + (1 - c.plasticity_floor) * np.exp(-self.age / c.plasticity_tau)

    # ------------------------------------------------------------------ propagation
    def forward(self, x: np.ndarray, prior: np.ndarray | None = None) -> dict:
        """Propagation. `prior` : biais moteur inné ajouté aux préférences d'action."""
        x = x.astype(F32, copy=False)
        z = self.W1 @ x + self.b1
        h = (np.tanh(np.maximum(z, 0.0)) * self.alive).astype(np.float64)
        # variabilité comportementale : les préférences apprises restent bornées
        logits = LOGIT_MAX * np.tanh((self.Wp @ h + self.bp) / LOGIT_MAX)
        if prior is not None:
            logits = logits + prior
        logits -= logits.max()
        pi = np.exp(logits)
        pi /= pi.sum()
        v = float(self.wv @ h + self.bv)
        return {"x": x, "z": z, "h": h, "pi": pi, "v": v}

    def act(self, c: dict) -> int:
        a = int(np.searchsorted(np.cumsum(c["pi"]), self.rng.random()))
        return min(a, N_ACTIONS - 1)

    def prediction_error(self, c: dict, a: int, x_next: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        u = np.zeros(self.H + N_ACTIONS, dtype=F32)
        u[: self.H] = c["h"]
        u[self.H + a] = 1.0
        err = self.Wf @ u - x_next
        return float(np.mean(err * err)), err, u

    @staticmethod
    def _nearest(memory: np.ndarray, n: int, h: np.ndarray, default: float) -> float:
        if n == 0:
            return default
        diff = memory[:n] - h.astype(F32)
        return float(np.sqrt(np.min(np.einsum("ij,ij->i", diff, diff))))

    def novelty(self, h: np.ndarray) -> float:
        """Distance de l'état cortical `h` au souvenir le plus proche (0 = déjà vu)."""
        return self._nearest(self.memory, self.memory_n, h, NOV_SCALE)

    def remember(self, h: np.ndarray) -> None:
        self.memory[self.memory_i] = h
        self.memory_i = (self.memory_i + 1) % len(self.memory)
        self.memory_n = min(self.memory_n + 1, len(self.memory))

    def object_novelty(self, x_what: np.ndarray, store: bool = True) -> float:
        """Nouveauté de l'identité d'un objet (voie du « quoi »).

        `x_what` ne contient que les traits de l'objet touché : la réponse corticale ne
        dépend donc que de ce que le cortex sait coder des objets.

        L'habituation est progressive : chaque souvenir a une force d'encodage qui croît à
        chaque nouvelle exposition ; un souvenir faible laisse subsister de la nouveauté.
        """
        # réponse évoquée = activité avec l'objet - activité spontanée (sans entrée)
        both = self.activations(np.stack([x_what, np.zeros_like(x_what)]))
        h = (both[0] - both[1]).astype(F32)
        n = self.obj_memory_n
        if n == 0:
            nov, j, d = 2 * OBJ_SCALE, -1, np.inf
        else:
            diff = self.obj_memory[:n] - h
            dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            nov = float(np.min(dist + 2 * OBJ_SCALE * (1.0 - self.obj_strength[:n])))
            j = int(np.argmin(dist))   # souvenir le plus ressemblant
            d = float(dist[j])
        if store:
            if j >= 0 and d < 0.6 * OBJ_SCALE:
                # ré-exposition : le souvenir se renforce (habituation)
                self.obj_strength[j] = min(1.0, self.obj_strength[j] + self.cfg.habituation_rate)
                self.obj_memory[j] += F32(0.5) * (h - self.obj_memory[j])
            else:
                i = self.obj_memory_i
                self.obj_memory[i] = h
                self.obj_strength[i] = self.cfg.habituation_rate
                self.obj_memory_i = (i + 1) % len(self.obj_memory)
                self.obj_memory_n = min(n + 1, len(self.obj_memory))
        return nov

    # ------------------------------------------------------------------ apprentissage
    def learn(self, c: dict, a: int, r_ext: float, c_next: dict, done: bool,
              learn_enabled: bool = True) -> dict:
        cfg = self.cfg
        p = self.plasticity()
        x, z, h, pi = c["x"], c["z"], c["h"], c["pi"]

        # --- modèle prédictif (erreur de prédiction sensorielle)
        pe, err, u = self.prediction_error(c, a, c_next["x"])
        # --- curiosité : nouveauté de la situation atteinte, relative à l'habitude
        #     (centrée : n'incite pas à prolonger artificiellement les épisodes)
        nov = self.novelty(c_next["h"])
        self.remember(c_next["h"])
        r_int = cfg.curiosity_coef * float(np.clip((nov - self.nov_ema) / NOV_SCALE, -1.0, 1.0))
        self.nov_ema += 0.01 * (nov - self.nov_ema)
        r = r_ext + r_int

        # --- erreur de prédiction de récompense (signal dopaminergique)
        v_next = 0.0 if done else c_next["v"]
        delta = float(np.clip(r + cfg.gamma * v_next - c["v"], -2.0, 2.0))

        if learn_enabled:
            gl = cfg.gamma * cfg.trace_lambda
            g_logits = self._policy_grad(h, pi, a)
            active = (z > 0) * (1.0 - h * h) * self.alive  # dérivée de tanh(relu(z))
            # traces d'éligibilité (étiquettes synaptiques)
            self.eWp *= gl
            self.eWp += np.outer(g_logits, h)
            self.ebp = gl * self.ebp + g_logits
            self.ewv = gl * self.ewv + h
            self.ebv = gl * self.ebv + 1.0
            dh = (self.Wp.T @ g_logits + self.wv) * active
            self.eW1 *= F32(gl)
            self.eW1 += np.outer(dh.astype(F32), x)

            # entropie (exploration)
            logp = np.log(pi + 1e-12)
            g_ent = -pi * (logp - (pi * logp).sum())

            # pas d'apprentissage normalisés par l'énergie de l'activité (stabilité)
            hh = 1.0 + float(h @ h)
            a_act, a_crit = p * cfg.lr_actor / hh, p * cfg.lr_critic / hh
            self.Wp += a_act * (delta * self.eWp + cfg.entropy_coef * np.outer(g_ent, h))
            self.bp += a_act * (delta * self.ebp + cfg.entropy_coef * g_ent)
            self.wv += a_crit * delta * self.ewv
            self.bv += a_crit * delta * self.ebv

            # apprentissage prédictif (non supervisé) — règle delta normalisée (NLMS)
            err_c = np.clip(err, -1.0, 1.0)
            self.Wf -= np.outer(err_c, u * F32(p * cfg.lr_forward / (1.0 + float(u @ u))))
            dh_f = (self.Wf[:, : self.H].T @ err_c) * active
            xx = float(x @ x)
            lr_body = cfg.lr_forward / (1.0 + xx)

            # mise à jour du cortex : renforcement (via les traces) + prédiction
            self.W1 += self.eW1 * F32(p * cfg.lr_body * delta / (1.0 + xx))
            self.W1 -= np.outer((p * lr_body * dh_f).astype(F32), x)
            np.multiply(self.W1, self.M, out=self.W1)
            self.b1 -= (p * lr_body * dh_f).astype(F32)

        # --- statistiques d'activité
        tau = cfg.coact_tau
        self.pre += tau * (x - self.pre)
        self.post += tau * (h - self.post)
        if self.age % COACT_EVERY == 0:
            k = tau * COACT_EVERY
            self.coact *= F32(1.0 - k)
            self.coact += np.outer((k * h).astype(F32), x)
        self.pe_ema += 0.01 * (pe - self.pe_ema)
        self.age += 1

        if done:
            self.end_episode()
        if learn_enabled and self.age % cfg.structural_period == 0:
            self.structural_plasticity()
        return {"pe": pe, "r_int": r_int, "delta": delta, "novelty": nov}

    def _policy_grad(self, h: np.ndarray, pi: np.ndarray, a: int) -> np.ndarray:
        g = -pi.copy()
        g[a] += 1.0
        raw = self.Wp @ h + self.bp
        return g * (1.0 - np.tanh(raw / LOGIT_MAX) ** 2)  # dérivée de la saturation

    # ------------------------------------------------------------------ réactivation hippocampique
    def replay(self, transitions: list, passes: int = 1) -> None:
        """Rejoue des expériences récentes en ordre inverse (Foster & Wilson, 2006).

        Pendant le repos, l'hippocampe réactive les séquences vécues, le plus souvent à
        rebours après une récompense : la valeur du but se propage ainsi jusqu'au départ en
        quelques essais. Seuls les « ganglions de la base » (acteur/critique) sont modifiés.
        """
        cfg = self.cfg
        p = self.plasticity()
        for _ in range(passes):
            for x, a, r, x2, done, prior in reversed(transitions):
                c = self.forward(x, prior)
                v2 = 0.0 if done else self.forward(x2)["v"]
                delta = float(np.clip(r + cfg.gamma * v2 - c["v"], -2.0, 2.0))
                h = c["h"]
                hh = 1.0 + float(h @ h)
                g = self._policy_grad(h, c["pi"], a)
                self.wv += p * cfg.lr_critic / hh * delta * h
                self.bv += p * cfg.lr_critic / hh * delta
                if cfg.replay_actor:
                    self.Wp += p * cfg.lr_actor / hh * delta * np.outer(g, h)
                    self.bp += p * cfg.lr_actor / hh * delta * g

    # ------------------------------------------------------------------ plasticité structurelle
    def structural_plasticity(self) -> None:
        cfg = self.cfg
        rng = self.rng
        p = self.plasticity()
        alive_rows = self.alive[:, None]

        # oubli passif des poids (équivalent à weight_decay appliqué à chaque étape)
        self.W1 *= F32((1.0 - cfg.weight_decay) ** cfg.structural_period)
        np.clip(self.W1, -3.0, 3.0, out=self.W1)

        # une entrée est « utilisée » si le neurone présynaptique décharge assez souvent
        pre_use = np.minimum(self.pre / cfg.pre_threshold, 1.0)          # (D,)
        used = pre_use >= 1.0

        # 1) élagage « use it or lose it » : les synapses faibles disparaissent, très
        #    probablement si leur entrée est silencieuse, rarement sinon (compétition)
        weak = self.M & (np.abs(self.W1) < cfg.prune_threshold)
        prob_prune = np.where(used, cfg.prune_prob_used, cfg.prune_prob)[None, :] * p
        prune = weak & (rng.random(self.M.shape) < prob_prune)
        self.M &= ~prune
        self.synapses_pruned += int(prune.sum())

        # 2) synaptogenèse dépendante de l'activité pré- ET postsynaptique
        post_use = np.minimum(self.post / cfg.target_rate, 1.0)          # (H,)
        prob_grow = cfg.grow_rate * p * np.outer(post_use, pre_use)
        grow = ~self.M & alive_rows & (rng.random(self.M.shape) < prob_grow)
        self.M |= grow
        self.W1[grow] = rng.normal(0.03, 0.02, int(grow.sum())).astype(F32)
        self.synapses_formed += int(grow.sum())

        # 3) capacité dendritique : on retire les synapses les plus faibles en excès
        fan_in = self.M.sum(1)
        for i in np.where(fan_in > cfg.max_fan_in)[0]:
            idx = np.where(self.M[i])[0]
            drop = idx[np.argsort(np.abs(self.W1[i, idx]))[: fan_in[i] - cfg.max_fan_in]]
            self.M[i, drop] = False
            self.synapses_pruned += len(drop)

        # 4) plasticité homéostatique : ramène l'activité moyenne vers la consigne
        alive_idx = np.where(self.alive)[0]
        ratio = cfg.target_rate / (self.post[alive_idx] + 1e-3)
        scale = np.clip(ratio ** 0.5, 1.0 - cfg.homeostasis, 1.0 + cfg.homeostasis)
        self.W1[alive_idx] *= scale[:, None]

        # 5) apoptose des neurones silencieux (après une période de grâce)
        silent = self.alive & (self.post < cfg.silent_threshold) & (self.age - self.birth > 2000)
        for i in np.where(silent)[0]:
            if self.alive.sum() <= cfg.min_neurons:
                break
            self._kill(i)

        # 6) neurogenèse déclenchée par la nouveauté
        novelty = self.pe_ema / PE_SCALE
        if novelty > cfg.novelty_threshold:
            prob = cfg.neurogenesis_rate * p * np.tanh(novelty - cfg.novelty_threshold)
            dormant = np.where(~self.alive)[0]
            if len(dormant) and rng.random() < prob:
                self._birth(int(dormant[0]))

        self.W1 *= self.M

    def _kill(self, i: int) -> None:
        self.alive[i] = False
        self.M[i] = False
        self.W1[i] = 0.0
        self.b1[i] = 0.0
        self.Wp[:, i] = 0.0
        self.wv[i] = 0.0
        self.Wf[:, i] = 0.0
        self.coact[i] = 0.0
        self.post[i] = 0.0
        self.deaths += 1

    def _birth(self, i: int) -> None:
        cfg = self.cfg
        # le nouveau neurone se connecte préférentiellement aux entrées actives
        w = self.pre + 1e-3
        k = max(1, int(cfg.initial_density * self.D))
        idx = self.rng.choice(self.D, size=k, replace=False, p=w / w.sum())
        self.M[i] = False
        self.M[i, idx] = True
        self.W1[i] = 0.0
        self.W1[i, idx] = self.rng.normal(0.0, 1.0 / np.sqrt(k), k).astype(F32)
        self.b1[i] = 0.05
        self.post[i] = cfg.silent_threshold * 3  # laisse au neurone le temps de s'intégrer
        self.alive[i] = True
        self.birth[i] = self.age
        self.births += 1

    def activations(self, X: np.ndarray) -> np.ndarray:
        """Activité corticale pour un lot d'observations (lignes de X)."""
        return np.tanh(np.maximum(X.astype(F32) @ self.W1.T + self.b1, 0.0)) * self.alive

    # ------------------------------------------------------------------ sauvegarde
    def state(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "M": self.M, "alive": self.alive, "Wp": self.Wp,
                "wv": self.wv, "pre": self.pre, "post": self.post}
