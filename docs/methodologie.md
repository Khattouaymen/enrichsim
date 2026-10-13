# EnrichSim — Méthodologie

*Document destiné à l'étudiante et à son encadrant·e. Il décrit ce que fait le modèle,
comment chaque concept de psychologie / neurosciences y est opérationnalisé, comment les
données sont analysées, et ce qu'il faut déclarer dans le mémoire.*

---

## 1. Question de recherche

Les travaux de Rosenzweig, Bennett et Diamond (1962-1972) ont montré que des rats élevés en
**environnement enrichi** (EC : grande cage, congénères, objets renouvelés) ont un cortex plus
épais et plus lourd, davantage de synapses et d'arborisation dendritique que des rats élevés
en **condition standard** (SC) ou **appauvrie** (IC : isolement, cage vide). Les effets
comportementaux (labyrinthe de Hebb-Williams, reconnaissance d'objet) existent mais sont plus
variables d'une étude à l'autre (van Praag, Kempermann & Gage, 2000 ; Simpson & Kelly, 2011).

Le projet pose la question suivante, sur un **modèle computationnel** :

> Un cerveau artificiel doté de règles de plasticité biologiquement plausibles, élevé dans des
> environnements plus ou moins riches, présente-t-il les mêmes différences structurelles et
> comportementales que l'animal ? Quelles composantes de l'enrichissement en sont responsables,
> et existe-t-il une période critique ?

Il s'agit d'une **simulation** : aucun animal n'est utilisé. Le modèle est un outil pour
tester la *suffisance* de mécanismes simples (plasticité dépendante de l'activité) à produire
les effets observés, pas une preuve de ce qui se passe chez le rat.

## 2. Correspondance concepts ↔ modèle

| Concept (psychologie / neurosciences) | Opérationnalisation dans EnrichSim | Fichier |
|---|---|---|
| Sujet expérimental (rat) | Un cerveau artificiel (`PlasticBrain`) dans un corps qui se déplace sur une grille | `brain.py`, `subject.py` |
| Cage d'élevage IC / SC / EC | Grille paramétrée : taille, objets, renouvellement, congénères, cloisons | `config.py`, `environment.py` |
| Perception (vision, vibrisses) | Voisinage local 5 × 5 : murs, nourriture, congénères, traits sensoriels des objets | `environment.py` |
| Cellules de lieu (hippocampe) | 16 cellules à champ gaussien ; carte différente pour chaque lieu (*remapping*, Muller & Kubie, 1987) | `environment.py` |
| Cortex | Couche de neurones (ReLU saturante) : 64 à la naissance, 112 au maximum | `brain.py` |
| Apprentissage par la récompense, dopamine | Acteur-critique ; l'erreur de prédiction de récompense δ joue le rôle du signal dopaminergique ; traces d'éligibilité | `brain.py` |
| Cerveau prédictif | Le cortex apprend à prédire la sensation suivante ; l'erreur de prédiction mesure la surprise | `brain.py` |
| Curiosité / exploration | Récompense intrinsèque = nouveauté de la situation par rapport à la mémoire récente | `brain.py` |
| Mémoire de reconnaissance (« quoi » / « où ») | Deux mémoires : situations (« où ») et objets touchés (« quoi », cortex périrhinal ; Brown & Aggleton, 2001) | `brain.py` |
| Réflexe d'orientation, habituation (Sokolov) | Un objet perçu comme nouveau prolonge l'investigation ; l'effet s'atténue à chaque exposition | `subject.py` |
| Alternance spontanée | Biais moteur inné à continuer tout droit, identique pour tous les sujets | `subject.py` |
| Synaptogenèse (« fire together, wire together ») | Création de synapses proportionnelle à la co-activité pré × post | `brain.py` |
| Élagage (« use it or lose it ») | Suppression des synapses faibles, surtout si leur entrée est silencieuse | `brain.py` |
| Neurogenèse | Recrutement d'un neurone quand l'erreur de prédiction reste élevée (nouveauté) | `brain.py` |
| Apoptose | Mort des neurones durablement silencieux | `brain.py` |
| *Synaptic scaling* (Turrigiano, 2008) | Homéostasie : l'activité de chaque neurone est ramenée vers une consigne | `brain.py` |
| Période critique | Option : toute la plasticité décroît exponentiellement avec l'âge (plancher 25 %) | `brain.py`, `config.py` |

### Mesures « neuroanatomiques »

| Mesure du modèle | Analogue histologique |
|---|---|
| Nombre de neurones | Neurogenèse nette |
| Nombre de synapses | Densité synaptique, « épaisseur corticale » |
| Synapses par neurone | Complexité dendritique (analyse de Sholl) |
| Force synaptique moyenne | Efficacité synaptique |
| Territoire « objets » | Part du cortex dédiée à une modalité (cartes corticales) |
| Dimensionnalité (ratio de participation) | Richesse des représentations |
| Séparation de patterns | Capacité à distinguer deux objets proches (gyrus denté ; Yassa & Stark, 2011) |
| Indice de discrimination **neuronal** (NOR) | Différence de réponse corticale objet nouveau / familier (analogue de l'imagerie ou de l'électrophysiologie) |

Les mesures anatomiques et représentationnelles sont prises **en fin d'élevage, avant les
tests**. Les mesures représentationnelles sont calculées sur un **jeu de stimuli de référence
identique pour tous les sujets**, composé d'objets jamais vus pendant l'élevage, afin de ne
favoriser aucune condition. Seul l'indice de discrimination neuronal est mesuré pendant le test
de reconnaissance d'objet, à partir des réponses du cortex aux deux objets.

### Batterie comportementale

Ordre fixe, du moins au plus exigeant, dans des arènes qu'aucun groupe n'a vues :

1. **Champ ouvert** (Hall, 1934) : exploration (part de l'arène visitée), temps au centre
   (indice inverse de thigmotaxie).
2. **Reconnaissance d'objet nouveau** (NOR ; Ennaceur & Delacour, 1988) : familiarisation avec
   deux objets identiques, puis l'un est remplacé ; indice de discrimination
   DI = (t_nouveau − t_familier) / (t_nouveau + t_familier). Quatre sessions avec des paires
   d'objets différentes, côté de l'objet nouveau contrebalancé. DI non calculé si l'exploration
   totale est inférieure à 5 contacts (critère usuel d'exclusion).
3. **Labyrinthe de Hebb-Williams** (Hebb & Williams, 1946 ; Rabinovitch & Rosvold, 1951) : quatre
   problèmes, huit essais chacun ; une erreur = un déplacement qui n'approche pas du but.
   Mesures : erreurs, latence, essais jusqu'au critère (≤ 2 erreurs). Six repères visuels (des
   objets encastrés dans les murs) aident à reconnaître les lieux.
4. **Apprentissage inversé** : le but du premier problème change de place ; erreurs et
   persévération (visites de l'ancien but) mesurent la flexibilité cognitive.

**Le cerveau continue d'apprendre pendant les tests**, comme l'animal. Dans le protocole
principal, sa structure continue aussi de changer (synapses, neurones). Or la batterie dure en
moyenne environ 5 800 étapes, soit près de 30 % de la durée de l'élevage, alors que chez le rat
les tests sont courts par rapport à des semaines d'hébergement. L'état du cerveau est donc
mesuré avant la batterie puis après chaque test (`test_development.csv`) ; voir la section 6.

## 3. Hypothèses (formulées avant l'expérience finale)

**Expérience 1 — IC / SC / EC (réplique de Rosenzweig)**
- H1a (structure) : EC > SC > IC pour le nombre de synapses, la complexité dendritique et le
  territoire cortical « objets ».
- H1b (représentations) : EC > IC pour la séparation de patterns et l'indice de discrimination
  neuronal.
- H1c (comportement) : EC fait moins d'erreurs au Hebb-Williams et discrimine mieux l'objet
  nouveau que IC. *Hypothèse plus incertaine : la littérature animale rapporte des effets
  comportementaux plus faibles et moins réguliers que les effets anatomiques.*

**Expérience 2 — Dose-réponse** (5 niveaux d'enrichissement entre IC et EC)
- H2 : relation monotone croissante entre le niveau d'enrichissement et les mesures
  structurelles, éventuellement avec un plafond (terme quadratique négatif).

**Expérience 3 — Période critique** (plan 2 × 2 : enrichi ou non pendant la 1re / la 2e moitié
de l'élevage ; plasticité décroissante avec l'âge)
- H3a : un effet principal de l'enrichissement précoce plus grand que celui de l'enrichissement
  tardif (η²p précoce > η²p tardif).
- H3b : un enrichissement tardif reste efficace mais à un moindre degré (plasticité résiduelle
  de l'adulte).

**Expérience 4 — Ablation** (EC privé d'une composante à la fois)
- H4 : les objets (complexité sensorielle) et la nouveauté portent l'essentiel de l'effet
  structurel ; l'espace et les congénères y contribuent moins. *Hypothèse exploratoire.*

## 4. Plan expérimental

- **Facteur** : environnement d'élevage (inter-sujets).
- **Effectif** : 20 sujets par groupe ; 360 sujets au total pour les quatre expériences, plus 60
  pour l'expérience de contrôle (section 6).
- **Durée d'élevage** : 20 000 étapes (une étape = un déplacement).
- **Plan apparié (« split-litter »)** : le sujet n° i de chaque groupe démarre avec exactement le
  même cerveau initial (même graine aléatoire), comme des petits d'une même portée répartis
  entre les conditions. Toute différence finale provient donc de l'environnement (et de la
  part aléatoire du comportement pendant l'élevage).
- **Graine de l'expérience finale : 2027**, différente des graines utilisées pendant les pilotes
  (2026). Les résultats rapportés sont donc indépendants des données ayant servi au calibrage.
- **Reproductibilité** : `metadata.json` enregistre tous les paramètres, les versions de Python et
  de NumPy et la graine ; la même commande produit exactement les mêmes données. L'expérience
  finale a d'ailleurs été relancée une seconde fois avec le code définitif : les données des
  quatre expériences sont identiques à la première exécution (vérification automatique, valeur
  par valeur).

```bash
pip install -r requirements.txt
python -m enrichsim run all --n 20 --steps 20000 --seed 2027 --out results
python -m enrichsim report results/main      # régénère un rapport
```

## 5. Analyses statistiques (plan fixé avant l'expérience finale)

Le plan est codé dans `enrichsim/stats.py`, commité **avant** le lancement de l'exécution
finale (voir l'historique git).

1. **Descriptives** : M, ET, IC 95 %, médiane par groupe.
2. **Conditions d'application** : Shapiro-Wilk par groupe et Levene (centré sur la médiane),
   α = .05.
3. **Test global** :
   - normalité et homogénéité → ANOVA de Fisher, post-hoc de Tukey HSD ;
   - normalité sans homogénéité → ANOVA de Welch, post-hoc de Games-Howell ;
   - normalité rejetée → Kruskal-Wallis, post-hoc de Mann-Whitney corrigés par Holm.
4. **Tailles d'effet** : η² et ω² (ε² pour Kruskal-Wallis) ; g de Hedges pour chaque paire.
   Repères de Cohen : 0,2 petit ; 0,5 moyen ; 0,8 grand.
5. **Comparaisons multiples** : les p globaux sont corrigés par Holm au sein de chaque famille
   (anatomie / comportement). Ce sont les p corrigés qui servent à conclure.
6. **Appariement** : un test de Friedman, qui exploite le plan apparié, est rapporté en
   complément ; l'analyse principale (groupes indépendants) est plus conservatrice.
7. **Courbes d'apprentissage** : modèle linéaire mixte
   `erreurs ~ groupe × essai + problème`, intercept et pente aléatoires par sujet (intercept seul
   si le modèle ne converge pas) ; test de Wald conjoint de l'interaction groupe × essai.
8. **Dose-réponse** : régression linéaire sur le niveau d'enrichissement (+ terme quadratique)
   et corrélation de Spearman.
9. **Période critique** : ANOVA 2 × 2 (type II), η² partiel ; contraste de Welch EC→IC vs IC→EC.
10. **Ablation** : test de Dunnett (chaque groupe vs EC) et « part de l'effet perdue »
    = (EC − groupe) / (EC − IC), calculée seulement quand EC diffère significativement de IC.
11. **NOR** : test t à un échantillon de l'indice de discrimination contre 0 dans chaque groupe.
    Attention : qu'un groupe soit significatif et un autre non ne prouve pas que les groupes
    diffèrent (Gelman & Stern, 2006). Seule la comparaison directe des groupes (point 3)
    permet de conclure à une différence.
12. **Valeurs de *p* et simulation** : dans une simulation, on peut rendre un *p* aussi petit qu'on
    veut en ajoutant des sujets. Les tailles d'effet et la forme des résultats comptent donc
    davantage que les seuils de significativité.

Les résultats sont présentés au format APA 7 : *F*(2, 57) = 3,95, *p* = ,025, η² = ,12.
*(Le rapport automatique utilise le point décimal ; le remplacer par la virgule dans le mémoire
si la norme de l'université l'exige.)*

## 6. Analyse complémentaire exploratoire — décidée après l'expérience finale

**Constat.** En examinant les cerveaux enregistrés, on a constaté après l'expérience finale que
la plasticité structurelle, restée active pendant les tests, rapproche fortement les cerveaux
des différents groupes : les tests (objets du test de reconnaissance, repères des labyrinthes)
« ré-enrichissent » les cerveaux IC, et les cerveaux EC perdent une partie de leurs synapses. Les
derniers tests (labyrinthe, inversion) comparent donc des cerveaux devenus plus semblables
qu'en fin d'élevage. Ce point n'avait pas été anticipé.

**Mesure.** L'état du cerveau est mesuré avant la batterie de tests puis après chaque test. On
calcule à chaque étape l'effet du groupe (η²) sur le nombre de synapses, le territoire
« objets », le nombre de neurones et la séparation de patterns.

**Expérience de contrôle (plan `control`).** Mêmes sujets, mêmes graines et donc même élevage que
l'expérience 1 : les cerveaux sont identiques en fin d'élevage (vérifié automatiquement). Seule
différence : pendant les tests, l'architecture du cerveau est figée (aucune synapse ni aucun
neurone n'apparaît ou ne disparaît). L'apprentissage des poids synaptiques, l'oubli passif et
l'homéostasie continuent.

**Analyses.** Pour chaque mesure prise pendant les tests :
- effet du groupe dans chaque protocole (η²) ;
- effet du protocole (architecture figée − plastique) : test *t* apparié, tous sujets confondus ;
- interaction groupe × protocole : comparaison des groupes sur la différence entre protocoles
  (avec deux protocoles mesurés sur les mêmes sujets, c'est le test de l'interaction), test
  choisi comme en section 5, *p* corrigés de Holm.

**Vérification du test de l'objet nouveau** (`scripts/nor_mecanisme.py`). Le script rejoue
l'élevage et les deux premiers tests de 8 sujets IC et 8 sujets EC de l'expérience 1 (mêmes
graines, donc mêmes sujets ; il vérifie que les indices de discrimination recalculés sont
identiques à ceux de `results/main/subjects.csv`). À chaque contact avec un objet pendant la
phase de test, il enregistre la nouveauté perçue, le bonus d'investigation qui en découle et
s'il atteint son plafond. Résultats dans `results/main/exploratoire_nor/`.

Ces analyses ont été décidées **après** avoir vu les résultats de l'expérience finale : elles
sont exploratoires et doivent être présentées comme telles. Le code d'analyse de l'expérience
de contrôle a été écrit et commité avant de disposer de ses résultats.

## 7. Phase de calibrage exploratoire — à déclarer

Avant l'expérience finale, le modèle a été mis au point sur des **pilotes** (graine 2026). Par
transparence, le mémoire doit indiquer que les modifications suivantes ont été faites au vu
des résultats des pilotes :

1. **Mémoire « quoi » / « où »** : ajout d'une mémoire des objets séparée de la mémoire des
   situations, pour que la reconnaissance d'objet ne soit pas noyée dans la nouveauté de
   l'ensemble de la scène.
2. **Habituation progressive** : la familiarité d'un objet augmente graduellement à chaque
   exposition (au lieu d'être acquise en une fois).
3. ***Remapping* des cellules de lieu** : chaque lieu reçoit sa propre carte, pour qu'une carte
   apprise dans la cage ne donne pas un avantage artificiel dans les labyrinthes de test.
4. **Format Hebb-Williams** : les labyrinthes initiaux ont été remplacés par des problèmes au
   format Hebb-Williams (arène carrée, barrières, départ et but dans des coins opposés) avec
   une cotation des erreurs par zones.
5. **Réactivation hippocampique (*replay*)** : testée, puis **désactivée** dans la version finale
   (`replay_enabled = False`) ; le code reste disponible pour une expérience complémentaire.
6. **Bug des congénères corrigé** : une erreur dans le déplacement des congénères a été
   corrigée.

Après ces changements, **tous les paramètres ont été figés** ; l'expérience finale utilise une
nouvelle graine. Les pilotes doivent être décrits comme exploratoires et leurs résultats ne
doivent pas être additionnés à ceux de l'expérience finale.

## 8. Limites

- **Raisonnement en partie circulaire sur l'anatomie.** La règle « use it or lose it » élague
  les synapses dont l'entrée est silencieuse. Dans IC, les entrées « objets » ne sont jamais
  actives : une partie de l'effet structurel (territoire « objets », nombre de synapses) est
  donc **attendue par construction**. Ce qui n'est pas imposé par la règle et constitue le vrai
  test du modèle : l'ampleur relative des effets selon les composantes (ablation), leur
  dépendance à l'âge (période critique), la forme de la relation dose-réponse, et surtout le
  **transfert** vers les représentations (séparation de patterns, discrimination neuronale) et
  le comportement.
- **Transfert comportemental faible.** Dans le pilote comme dans l'expérience finale, les
  différences anatomiques massives ne se traduisent que faiblement en comportement. C'est un
  résultat en soi, à discuter : (a) la littérature animale rapporte aussi des effets
  comportementaux plus modestes et variables ; (b) le modèle est très petit (≈ 100 neurones) et
  les tâches sont courtes ; (c) une « plus grande » structure n'est pas automatiquement une
  structure mieux utilisée ; (d) la plasticité restée active pendant les tests rapproche les
  cerveaux des groupes (section 6).
- **Plasticité pendant les tests.** La batterie de tests est longue (≈ 30 % de la durée de
  l'élevage) et modifie le cerveau. Les mesures anatomiques sont prises avant les tests, mais les
  derniers tests évaluent des cerveaux déjà transformés par les premiers.
- **Sujets partagés.** Les groupes EC et IC des expériences 1 et 4 sont les mêmes sujets simulés,
  et l'expérience de contrôle reprend exactement l'élevage de l'expérience 1 : ce ne sont pas des
  réplications indépendantes.
- **Simplicité du modèle** : pas de neurotransmetteurs distincts, pas d'hormones (stress de
  l'isolement), pas d'interactions sociales réelles (les congénères bougent au hasard), pas de
  sommeil.
- **Validité externe** : une simulation teste la cohérence interne d'une théorie ; elle ne
  remplace pas l'expérimentation animale ni humaine.
- **Paramètres libres** : de nombreux paramètres ont été fixés à des valeurs raisonnables sans
  ajustement systématique ; une analyse de sensibilité serait une extension utile.

## 9. Références

- Brown, M. W., & Aggleton, J. P. (2001). Recognition memory: What are the roles of the perirhinal cortex and hippocampus? *Nature Reviews Neuroscience, 2*(1), 51-61.
- Dember, W. N., & Fowler, H. (1958). Spontaneous alternation behavior. *Psychological Bulletin, 55*(6), 412-428.
- Diamond, M. C., Krech, D., & Rosenzweig, M. R. (1964). The effects of an enriched environment on the histology of the rat cerebral cortex. *Journal of Comparative Neurology, 123*(1), 111-119.
- Ennaceur, A., & Delacour, J. (1988). A new one-trial test for neurobiological studies of memory in rats. *Behavioural Brain Research, 31*(1), 47-59.
- Gao, P., & Ganguli, S. (2015). On simplicity and complexity in the brave new world of large-scale neuroscience. *Current Opinion in Neurobiology, 32*, 148-155.
- Gelman, A., & Stern, H. (2006). The difference between "significant" and "not significant" is not itself statistically significant. *The American Statistician, 60*(4), 328-331.
- Hall, C. S. (1934). Emotional behavior in the rat. *Journal of Comparative Psychology, 18*(3), 385-403.
- Hebb, D. O. (1949). *The organization of behavior*. Wiley.
- Hebb, D. O., & Williams, K. (1946). A method of rating animal intelligence. *Journal of General Psychology, 34*(1), 59-65.
- Muller, R. U., & Kubie, J. L. (1987). The effects of changes in the environment on the spatial firing of hippocampal complex-spike cells. *Journal of Neuroscience, 7*(7), 1951-1968.
- Nithianantharajah, J., & Hannan, A. J. (2006). Enriched environments, experience-dependent plasticity and disorders of the nervous system. *Nature Reviews Neuroscience, 7*(9), 697-709.
- Rabinovitch, M. S., & Rosvold, H. E. (1951). A closed-field intelligence test for rats. *Canadian Journal of Psychology, 5*(3), 122-128.
- Rosenzweig, M. R., Bennett, E. L., & Diamond, M. C. (1972). Brain changes in response to experience. *Scientific American, 226*(2), 22-29.
- Simpson, J., & Kelly, J. P. (2011). The impact of environmental enrichment in laboratory rats: Behavioural and neurochemical aspects. *Behavioural Brain Research, 222*(1), 246-264.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2e éd.). MIT Press.
- Turrigiano, G. G. (2008). The self-tuning neuron: Synaptic scaling of excitatory synapses. *Cell, 135*(3), 422-435.
- van Praag, H., Kempermann, G., & Gage, F. H. (2000). Neural consequences of environmental enrichment. *Nature Reviews Neuroscience, 1*(3), 191-198.
- Yassa, M. A., & Stark, C. E. L. (2011). Pattern separation in the hippocampus. *Trends in Neurosciences, 34*(10), 515-525.

*Vérifier chaque référence (pages, volumes) avant de l'insérer dans le mémoire.*
