# EnrichSim

Modèle computationnel des effets de l'environnement d'élevage (appauvri / standard / enrichi)
sur le développement cérébral et le comportement — projet de PFE (Master, psychologie).

Un cerveau artificiel à plasticité synaptique et structurelle (synaptogenèse, élagage,
neurogenèse, homéostasie) se développe dans une cage IC, SC ou EC (protocole de Rosenzweig),
puis passe une batterie de tests inspirés des paradigmes chez le rongeur : champ ouvert,
reconnaissance d'objet nouveau, labyrinthe de Hebb-Williams, apprentissage inversé.

Cinq expériences : `main` (IC / SC / EC), `dose` (dose-réponse), `timing` (période critique,
plan 2 × 2), `ablation` (rôle de chaque composante) et `control` (contrôle exploratoire : même
élevage que `main`, architecture du cerveau figée pendant les tests).

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
# test rapide (~1 min)
python -m enrichsim run main --n 3 --steps 3000 --out test_rapide --no-report

# expérience finale du mémoire : les 5 plans, 420 sujets
# (graine différente des pilotes ; environ 30 min sur 4 cœurs)
python -m enrichsim run all --n 20 --steps 20000 --seed 2027 --out results --workers 4

# (re)générer le rapport HTML d'une expérience
python -m enrichsim report results/main

# tableau de bord pour la soutenance
streamlit run app/dashboard.py

# tests
python -m pytest
```

## Contenu

| Élément | Rôle |
|---|---|
| `enrichsim/` | modèle (cerveau, environnements, tests comportementaux, expériences) |
| `enrichsim/stats.py` | plan d'analyse statistique (fixé avant l'expérience finale) et analyses exploratoires ajoutées ensuite |
| `enrichsim/report.py` | rapport HTML en français, résultats au format APA |
| `app/dashboard.py` | tableau de bord Streamlit |
| `docs/methodologie.md` | méthodologie, hypothèses, phase de calibrage, limites, références |
| `tests/` | tests pytest |

Chaque dossier de `results/` contient :

| Fichier | Contenu |
|---|---|
| `subjects.csv` | une ligne par sujet : anatomie en fin d'élevage et scores aux tests |
| `development.csv` | évolution du cerveau pendant l'élevage |
| `test_development.csv` | état du cerveau avant la batterie de tests puis après chaque test |
| `learning_curves.csv` | erreurs et latences essai par essai (labyrinthe, inversion) |
| `brains/*.npz` | cerveaux des 2 premiers sujets de chaque groupe, en fin d'élevage (`<groupe>_s<i>.npz`) et après les tests (`<groupe>_s<i>_apres_tests.npz`) |
| `metadata.json` | tous les paramètres (reproductibilité) |
| `stats/*.csv`, `rapport.html` | analyses statistiques et rapport |

À graine égale, les données sont identiques d'une exécution à l'autre (seule la durée de calcul
`runtime_s` varie).
