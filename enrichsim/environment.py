"""Monde en grille 2D : cages d'élevage et arènes de test.

L'agent perçoit un voisinage local (vision / vibrisses) et reçoit l'activité de cellules
de lieu (codage hippocampique de la position). Chaque case peut contenir un mur, de la
nourriture, un objet (décrit par un vecteur de traits sensoriels) ou un congénère.
"""
from __future__ import annotations

import zlib
from collections import deque
from functools import lru_cache

import numpy as np

from .config import BrainConfig, EnvConfig

EMPTY, WALL, FOOD, OBJECT = 0, 1, 2, 3
MOVES = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])  # N, E, S, O
N_ACTIONS = 4

NOVEL_OBJECT_BASE = 100_000  # identifiants d'objets jamais vus pendant l'élevage


@lru_cache(maxsize=None)
def object_features(type_id: int, n_features: int) -> np.ndarray:
    """Traits sensoriels (forme, texture, odeur...) d'un type d'objet.

    Déterministe : un même type d'objet est perçu de la même façon par tous les sujets.
    """
    rng = np.random.default_rng(7919 * (type_id + 1))
    feats = rng.uniform(0.0, 1.0, n_features)
    feats[rng.random(n_features) < 0.4] = 0.0  # traits parcimonieux
    if not feats.any():
        feats[rng.integers(n_features)] = 1.0
    feats.setflags(write=False)
    return feats


def obs_dim(brain: BrainConfig) -> int:
    side = 2 * brain.view_radius + 1
    return side * side * (3 + brain.object_features) + brain.place_cells ** 2


class GridWorld:
    """Monde en grille générique (cage ou arène)."""

    def __init__(self, height: int, width: int, brain: BrainConfig, rng: np.random.Generator,
                 room: str = "room"):
        self.h, self.w = height, width
        self.room = room  # identité du lieu : détermine la carte des cellules de lieu
        self.brain = brain
        self.rng = rng
        self.grid = np.full((height, width), EMPTY, dtype=np.int8)
        self.obj_type = np.full((height, width), -1, dtype=np.int64)
        self.agent = np.array([0, 0])
        self.mates: list[np.ndarray] = []
        self.food_respawn = True
        self.t = 0
        self._cache_key = b""
        self._cache = None
        self._build_place_cells()

    # ------------------------------------------------------------------ perception
    def _build_place_cells(self) -> None:
        p = self.brain.place_cells
        c = (np.arange(p) + 0.5) / p
        centers = np.array([(a, b) for a in c for b in c])
        # « remapping global » (Muller & Kubie, 1987) : chaque lieu reçoit sa propre
        # affectation des cellules de lieu ; une carte apprise ne se transfère pas ailleurs
        perm = np.random.default_rng(zlib.crc32(self.room.encode())).permutation(len(centers))
        self._pc_centers = centers[perm]
        self._pc_sigma2 = 2 * (1.0 / p) ** 2

    def _sensory_map(self) -> np.ndarray:
        """Carte sensorielle du monde entier (bordée de murs), recalculée si le décor change."""
        key = self.grid.tobytes() + self.obj_type.tobytes()
        if key != self._cache_key:
            r = self.brain.view_radius
            F = self.brain.object_features
            T = np.zeros((self.h + 2 * r, self.w + 2 * r, 3 + F), dtype=np.float32)
            T[..., 0] = 1.0  # l'extérieur est perçu comme un mur
            inner = T[r:r + self.h, r:r + self.w]
            inner[..., 0] = self.grid == WALL
            inner[..., 1] = self.grid == FOOD
            for y, x in np.argwhere(self.grid == OBJECT):
                inner[y, x, 3:] = object_features(int(self.obj_type[y, x]), F)
            self._cache, self._cache_key = T, key
        return self._cache

    def observe(self) -> np.ndarray:
        r = self.brain.view_radius
        side = 2 * r + 1
        ay, ax = int(self.agent[0]), int(self.agent[1])
        view = self._sensory_map()[ay:ay + side, ax:ax + side].copy()
        for m in self.mates:
            dy, dx = int(m[0]) - ay + r, int(m[1]) - ax + r
            if 0 <= dy < side and 0 <= dx < side:
                view[dy, dx, 2] = 1.0
        pos = np.array([(ay + 0.5) / self.h, (ax + 0.5) / self.w])
        place = np.exp(-((self._pc_centers - pos) ** 2).sum(1) / self._pc_sigma2)
        return np.concatenate([view.ravel(), place.astype(np.float32)])

    def object_view(self, y: int, x: int) -> np.ndarray:
        """Entrée de la voie du « quoi » : uniquement les traits de l'objet en (y, x).

        Quand l'animal explore un objet, il lui fait face : l'objet est toujours présenté
        à la même place (juste devant), ce qui rend la représentation indépendante du
        côté par lequel on l'aborde (invariance de la voie ventrale)."""
        r = self.brain.view_radius
        F = self.brain.object_features
        side = 2 * r + 1
        view = np.zeros((side, side, 3 + F), dtype=np.float32)
        if self.grid[y, x] == OBJECT:
            view[r, r + 1, 3:] = object_features(int(self.obj_type[y, x]), F)
        return np.concatenate([view.ravel(), np.zeros(self.brain.place_cells ** 2, np.float32)])

    # ------------------------------------------------------------------ dynamique
    def passable(self, y: int, x: int) -> bool:
        return 0 <= y < self.h and 0 <= x < self.w and self.grid[y, x] in (EMPTY, FOOD)

    def step(self, action: int) -> tuple[float, dict]:
        """Déplace l'agent. Retourne (récompense externe, infos)."""
        self.t += 1
        target = self.agent + MOVES[action]
        ty, tx = int(target[0]), int(target[1])
        info = {"touched_object": -1, "ate": False, "bumped": False}
        reward = 0.0
        occupied = any((m == target).all() for m in self.mates)
        if self.passable(ty, tx) and not occupied:
            self.agent = target
            if self.grid[ty, tx] == FOOD:
                reward = 1.0
                info["ate"] = True
                self.grid[ty, tx] = EMPTY
                if self.food_respawn:
                    self.place_random(FOOD)
        else:
            info["bumped"] = True
            if 0 <= ty < self.h and 0 <= tx < self.w and self.grid[ty, tx] == OBJECT:
                info["touched_object"] = int(self.obj_type[ty, tx])
                info["object_pos"] = (ty, tx)
        self._move_mates()
        self.on_step()
        return reward, info

    def on_step(self) -> None:  # surchargé par les cages (nouveauté)
        pass

    def _move_mates(self) -> None:
        for i, m in enumerate(self.mates):
            if self.rng.random() < 0.5:
                continue
            nxt = m + MOVES[self.rng.integers(N_ACTIONS)]
            if (self.passable(*nxt) and not (nxt == self.agent).all()
                    and not any((o == nxt).all() for j, o in enumerate(self.mates) if j != i)):
                self.mates[i] = nxt

    # ------------------------------------------------------------------ construction
    def empty_cells(self) -> np.ndarray:
        free = self.grid == EMPTY
        free[self.agent[0], self.agent[1]] = False
        for m in self.mates:
            free[m[0], m[1]] = False
        return np.argwhere(free)

    def place_random(self, kind: int, type_id: int = -1) -> tuple[int, int] | None:
        cells = self.empty_cells()
        if len(cells) == 0:
            return None
        y, x = cells[self.rng.integers(len(cells))]
        self.grid[y, x] = kind
        if kind == OBJECT:
            self.obj_type[y, x] = type_id
        return int(y), int(x)

    def connected(self) -> bool:
        """Toutes les cases franchissables sont-elles accessibles ?"""
        free = np.argwhere(np.isin(self.grid, (EMPTY, FOOD)))
        if len(free) == 0:
            return False
        seen = {tuple(free[0])}
        q = deque([tuple(free[0])])
        while q:
            y, x = q.popleft()
            for dy, dx in MOVES:
                n = (y + dy, x + dx)
                if n not in seen and self.passable(*n):
                    seen.add(n)
                    q.append(n)
        return len(seen) == len(free)

    def shortest_path(self, start, goal) -> int:
        """Distance BFS (murs et objets infranchissables). -1 si inaccessible."""
        start, goal = tuple(start), tuple(goal)
        dist = {start: 0}
        q = deque([start])
        while q:
            c = q.popleft()
            if c == goal:
                return dist[c]
            for dy, dx in MOVES:
                n = (c[0] + dy, c[1] + dx)
                if n not in dist and (self.passable(*n) or n == goal):
                    dist[n] = dist[c] + 1
                    q.append(n)
        return -1

    def render(self) -> np.ndarray:
        """Grille codée pour l'affichage : 0 vide, 1 mur, 2 nourriture, 3 objet, 4 congénère, 5 agent."""
        img = self.grid.astype(np.int16).copy()
        for m in self.mates:
            img[m[0], m[1]] = 4
        img[self.agent[0], self.agent[1]] = 5
        return img


class HomeCage(GridWorld):
    """Cage d'élevage paramétrée par une EnvConfig (IC, SC, EC, ou intermédiaire)."""

    def __init__(self, cfg: EnvConfig, brain: BrainConfig, rng: np.random.Generator,
                 type_offset: int = 0):
        super().__init__(cfg.size, cfg.size, brain, rng, room=f"cage-{cfg.name}")
        self.cfg = cfg
        self.type_offset = type_offset  # permet de créer des mondes aux objets inédits
        self.rearrangements = 0
        self._layout()

    def _layout(self) -> None:
        cfg = self.cfg
        for _ in range(50):
            self.grid[:] = EMPTY
            self.obj_type[:] = -1
            for _ in range(cfg.inner_walls):
                self._random_wall_segment()
            if self.connected():
                break
        else:
            self.grid[:] = EMPTY
        self.agent = self.empty_cells()[self.rng.integers(len(self.empty_cells()))]
        self.mates = []
        for _ in range(cfg.n_conspecifics):
            cells = self.empty_cells()
            self.mates.append(cells[self.rng.integers(len(cells))])
        if cfg.n_objects and cfg.object_pool:
            types = self.rng.choice(cfg.object_pool, size=cfg.n_objects,
                                    replace=cfg.n_objects > cfg.object_pool)
            for t in types:
                self.place_random(OBJECT, int(t) + self.type_offset)
            if not self.connected():  # un objet ne doit jamais isoler une zone
                self.grid[self.grid == OBJECT] = EMPTY
                self.obj_type[:] = -1
        n_food = max(1, int(round(cfg.food_density * cfg.size * cfg.size)))
        for _ in range(n_food):
            self.place_random(FOOD)

    def _random_wall_segment(self) -> None:
        length = int(self.rng.integers(2, 5))
        horizontal = self.rng.random() < 0.5
        y = int(self.rng.integers(1, self.h - 1))
        x = int(self.rng.integers(1, self.w - 1))
        for k in range(length):
            yy, xx = (y, x + k) if horizontal else (y + k, x)
            if 0 < yy < self.h - 1 and 0 < xx < self.w - 1:
                self.grid[yy, xx] = WALL

    def on_step(self) -> None:
        p = self.cfg.novelty_period
        if p and self.t % p == 0:
            # réaménagement : nouveaux objets, nouvelle disposition (protocole EC classique)
            agent, mates = self.agent.copy(), [m.copy() for m in self.mates]
            self._layout()  # nouveaux murs, objets, nourriture ; congénères replacés
            if self.passable(*agent) and not any((m == agent).all() for m in self.mates):
                self.agent = agent
            # les congénères restent où ils étaient quand c'est possible
            kept = []
            for old, new in zip(mates, self.mates):
                ok = self.passable(*old) and not (old == self.agent).all()                     and not any((old == k).all() for k in kept)
                kept.append(old if ok else new)
            self.mates = kept
            self.rearrangements += 1


# ---------------------------------------------------------------------------- arènes de test
MAZES = [
    # Problèmes de type Hebb-Williams (Hebb & Williams, 1946 ; Rabinovitch & Rosvold, 1951) :
    # arène carrée, départ (S) et but (G) dans des coins opposés, barrières (#) différentes
    # pour chaque problème. Aucune de ces configurations n'est vue pendant l'élevage.
    [
        "#########",
        "#......G#",
        "#.......#",
        "#.......#",
        "#.#####.#",
        "#.......#",
        "#.......#",
        "#S......#",
        "#########",
    ],
    [
        "#########",
        "#......G#",
        "#.......#",
        "#..######",
        "#.......#",
        "######..#",
        "#.......#",
        "#S......#",
        "#########",
    ],
    [
        "#########",
        "#.....#G#",
        "#.###.#.#",
        "#.#...#.#",
        "#.#.###.#",
        "#.......#",
        "#.#####.#",
        "#S......#",
        "#########",
    ],
    [
        "#########",
        "#......G#",
        "#.####..#",
        "#....#..#",
        "#.##.#.##",
        "#..#...##",
        "##.#.#..#",
        "#S.....##",
        "#########",
    ],
]
LANDMARK_BASE = 200_000  # repères visuels des labyrinthes (jamais vus pendant l'élevage)
N_LANDMARKS = 6
REVERSAL_GOAL = [(1, 1), (7, 7), (1, 1), (1, 1)]  # nouveau but pour l'apprentissage inversé


class Maze(GridWorld):
    def __init__(self, problem: int, brain: BrainConfig, rng: np.random.Generator,
                 goal: tuple[int, int] | None = None):
        layout = MAZES[problem]
        super().__init__(len(layout), len(layout[0]), brain, rng, room=f"maze-{problem}")
        self.food_respawn = False
        for y, row in enumerate(layout):
            for x, ch in enumerate(row):
                if ch == "#":
                    self.grid[y, x] = WALL
                elif ch == "S":
                    self.start = (y, x)
                elif ch == "G":
                    self.default_goal = (y, x)
        self.goal = goal or self.default_goal
        self._place_landmarks(problem)
        self.reset()

    def _place_landmarks(self, problem: int) -> None:
        """Repères visuels (objets) encastrés dans les murs, comme les indices visuels d'une
        salle d'expérience. Ils ne modifient pas les trajets mais aident à reconnaître les lieux."""
        rng = np.random.default_rng(1000 + problem)
        cand = []
        for y in range(self.h):
            for x in range(self.w):
                if self.grid[y, x] == WALL and any(
                        self.passable(y + dy, x + dx) for dy, dx in MOVES):
                    cand.append((y, x))
        for k, i in enumerate(rng.choice(len(cand), size=min(N_LANDMARKS, len(cand)), replace=False)):
            y, x = cand[i]
            self.grid[y, x] = OBJECT
            self.obj_type[y, x] = LANDMARK_BASE + 10 * problem + k

    def reset(self) -> None:
        self.grid[self.grid == FOOD] = EMPTY
        self.grid[self.goal] = FOOD
        self.agent = np.array(self.start)
        self.optimal = self.shortest_path(self.start, self.goal)
        if not hasattr(self, "goal_distance"):
            # distance réelle (en contournant les barrières) de chaque case au but
            dist = {tuple(self.goal): 0}
            q = deque([tuple(self.goal)])
            while q:
                c = q.popleft()
                for dy, dx in MOVES:
                    n = (c[0] + dy, c[1] + dx)
                    if n not in dist and self.passable(*n):
                        dist[n] = dist[c] + 1
                        q.append(n)
            self.goal_distance = dist


class OpenField(GridWorld):
    """Champ ouvert : exploration et thigmotaxie (anxiété)."""

    def __init__(self, size: int, brain: BrainConfig, rng: np.random.Generator):
        super().__init__(size, size, brain, rng, room="open-field")
        self.food_respawn = False
        self.agent = np.array([0, 0])  # départ dans un coin, comme en laboratoire


class ObjectArena(GridWorld):
    """Arène de reconnaissance d'objet nouveau (Ennaceur & Delacour, 1988)."""

    def __init__(self, size: int, brain: BrainConfig, rng: np.random.Generator,
                 left_type: int, right_type: int):
        super().__init__(size, size, brain, rng, room="object-arena")
        self.food_respawn = False
        self.left = (size // 2, 1)
        self.right = (size // 2, size - 2)
        self.set_objects(left_type, right_type)
        self.agent = np.array([size - 1, size // 2])

    def set_objects(self, left_type: int, right_type: int) -> None:
        for pos, t in ((self.left, left_type), (self.right, right_type)):
            self.grid[pos] = OBJECT
            self.obj_type[pos] = t
