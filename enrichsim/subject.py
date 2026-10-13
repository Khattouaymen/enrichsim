"""Un sujet expérimental : un cerveau qui se développe dans une cage puis passe les tests."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .brain import OBJ_SCALE, PlasticBrain
from .config import BrainConfig, Phase
from .environment import N_ACTIONS, GridWorld, HomeCage, obs_dim
from .metrics import brain_metrics


def motor_prior(last_action: int | None, bumped: bool, strength: float) -> np.ndarray:
    """Persistance motrice innée : tendance à poursuivre tout droit et à ne pas faire
    demi-tour (cf. alternance spontanée, Dember & Fowler, 1958). Identique pour tous."""
    prior = np.zeros(N_ACTIONS)
    if last_action is None or bumped or strength == 0:
        return prior
    prior[last_action] += strength
    prior[(last_action + 2) % N_ACTIONS] -= strength
    return prior


@dataclass
class EpisodeStats:
    steps: int = 0
    food: int = 0
    touches: int = 0
    objects_touched: set = field(default_factory=set)
    visited: set = field(default_factory=set)
    pe_sum: float = 0.0
    reached: bool = False


class Subject:
    def __init__(self, sid: int, brain_cfg: BrainConfig, seed: int):
        self.sid = sid
        self.rng = np.random.default_rng(seed)
        self.brain = PlasticBrain(brain_cfg, obs_dim(brain_cfg), self.rng)

    # ------------------------------------------------------------------ boucle perception-action
    def run(self, world: GridWorld, steps: int, stop_on_food: bool = False,
            on_step=None, learn: bool = True, replay: int = 0, replay_passes: int = 1) -> EpisodeStats:
        """Fait vivre le sujet dans `world` pendant `steps` étapes (apprentissage en ligne).

        `replay` : nombre de transitions récentes réactivées à la fin (0 = aucune).
        """
        brain = self.brain
        st = EpisodeStats()
        memory: list = []
        strength = brain.cfg.motor_persistence
        prior = motor_prior(None, False, strength)
        c = brain.forward(world.observe(), prior)
        for _ in range(steps):
            a = brain.act(c)
            r, info = world.step(a)
            done = stop_on_food and info["ate"]
            next_prior = motor_prior(a, info["bumped"], strength)
            x_next = world.observe()
            obj_nov = None
            if info["touched_object"] >= 0:
                # réflexe d'orientation (Sokolov) : un objet perçu comme nouveau retient
                # l'exploration, un objet familier est vite délaissé (habituation)
                obj_nov = brain.object_novelty(world.object_view(*info["object_pos"]))
                next_prior[a] += brain.cfg.investigation * min(obj_nov / OBJ_SCALE, 2.0)
            c_next = brain.forward(x_next, next_prior)
            out = brain.learn(c, a, r, c_next, done, learn_enabled=learn)
            out["object_novelty"] = obj_nov
            if replay:
                memory.append((c["x"], a, r + out["r_int"], c_next["x"], done, prior))
            prior = next_prior
            st.steps += 1
            st.pe_sum += out["pe"]
            st.food += info["ate"]
            if info["touched_object"] >= 0:
                st.touches += 1
                st.objects_touched.add(info["touched_object"])
            st.visited.add((int(world.agent[0]), int(world.agent[1])))
            if on_step is not None:
                on_step(world, info, out)
            c = c_next
            if done:
                st.reached = True
                break
        brain.end_episode()
        # réactivation déclenchée par la récompense (Ambrose, Pfeiffer & Foster, 2016)
        if replay and learn and brain.cfg.replay_enabled and st.food > 0:
            brain.replay(memory[-replay:], replay_passes)
        return st

    # ------------------------------------------------------------------ développement
    def develop(self, phases: list[Phase], log_every: int = 1000) -> list[dict]:
        """Phase d'élevage. Retourne la trajectoire développementale (mesures régulières)."""
        log: list[dict] = [{"age": 0, "phase": 0, "env": phases[0].env.name, **brain_metrics(self.brain)}]
        for k, phase in enumerate(phases):
            cage = HomeCage(phase.env, self.brain.cfg, self.rng)
            done = 0
            while done < phase.steps:
                chunk = min(log_every, phase.steps - done)
                st = self.run(cage, chunk, replay=self.brain.cfg.sleep_replay)
                done += chunk
                log.append({
                    "age": self.brain.age, "phase": k, "env": phase.env.name,
                    "food_rate": st.food / st.steps,
                    "touch_rate": st.touches / st.steps,
                    "distinct_objects": len(st.objects_touched),
                    "exploration": len(st.visited) / max(1, int((cage.grid != 1).sum())),
                    "prediction_error": st.pe_sum / st.steps,
                    **brain_metrics(self.brain),
                })
        return log
