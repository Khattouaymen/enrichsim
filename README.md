# EnrichSim

Modèle computationnel des effets de l'environnement d'élevage (appauvri / standard / enrichi)
sur le développement cérébral et le comportement — projet de PFE (Master, psychologie).

Un cerveau artificiel à plasticité synaptique et structurelle (synaptogenèse, élagage,
neurogenèse, homéostasie) se développe dans une cage IC, SC ou EC (protocole de Rosenzweig),
puis passe une batterie de tests inspirés des paradigmes chez le rongeur : champ ouvert,
reconnaissance d'objet nouveau, labyrinthe de Hebb-Williams, apprentissage inversé.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
# test rapide (~1 min)
python -m enrichsim run main --n 3 --steps 3000 --out test_rapide --no-report

# les 4 expériences (main, dose, timing, ablation)
python -m enrichsim run all --n 20 --steps 20000 --out results
```

Travail en cours : analyse statistique, rapport, tableau de bord Streamlit, tests, documentation.
