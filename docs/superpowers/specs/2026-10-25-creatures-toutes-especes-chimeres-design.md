# Conception : créatures de toutes espèces, visages et chimères pondérées

**Date** : 2026-10-25
**Auteur** : MissawB
**Statut** : à valider par l'utilisateur

---

## 1. Intention

Le projet génère aujourd'hui des quadrupèdes synthétiques (cylindres autour d'un squelette fixe de 35 articulations, 5 familles, 2048 points). Il doit devenir une **machine à créatures** :

- n'importe quel animal, réel ou imaginaire : mammifères, oiseaux, reptiles, poissons, insectes, dragons, monstres, mais aussi humanoïdes, personnages et jouets ;
- avec un **visage lisible** (museau ou bec, yeux, oreilles, mâchoire) et des détails (ailes, queue, cornes) ;
- et, comme finalité, des **chimères pondérées** : on choisit deux ou trois parents et le pourcentage de chacun (par ex. 70 % lion, 30 % dragon), ou on laisse le hasard choisir ; l'enfant se forme en direct dans la visionneuse, du bruit à la créature.

Le critère de l'utilisateur : la créature doit être **une création originale mais cohérente**. Les traits des parents se **fondent** les uns dans les autres ; ce n'est pas un assemblage de morceaux recollés.

## 2. Objectifs, non-objectifs, anti-objectifs

**Objectifs**
- Entraîner sur de vraies formes de créatures, gratuites, avec squelette.
- Un modèle qui accepte des squelettes de topologie quelconque (nombre d'os et branchements variables).
- Un nuage nettement plus dense que 2048 points, assez pour lire un visage.
- Des temps d'entraînement courts : c'est une priorité de conception, pas une optimisation tardive.
- Une visionneuse avec catalogue de toutes les créatures, curseurs de pourcentage, bouton aléatoire et génération en direct.

**Non-objectifs (pour l'instant)**
- Produire un maillage. La sortie reste un nuage de points ; un maillage pourra venir plus tard, sans que l'architecture en dépende.
- L'animation (mouvement) des créatures.
- La couleur ou la texture.

**Anti-objectifs** (le projet échoue même s'il « marche » techniquement)
- Une chimère qui ressemble à un patchwork, avec des coutures visibles entre les parents.
- Un corps qui ne tient pas debout, ou un visage illisible (une boule à la place de la tête).
- Un mélange qui ignore les pourcentages, par exemple 90 % lion et 10 % dragon qui donnerait un résultat à moitié-moitié.
- Un entraînement qui ne tient que sur plusieurs jours de GPU dès le départ.

## 3. Contraintes

- **Aucun achat.** Données, outils et calcul gratuits.
- **Calcul** : PC portable Windows 11, RTX 4060 Laptop 8 Go, Python 3.12 dans `.venv` (jamais le Python système). Kaggle (environ 30 h de GPU par semaine, sessions de 12 h) et Google Colab servent de renfort. Un entraînement doit donc pouvoir reprendre depuis un checkpoint (c'est déjà le cas dans `pipeline/trainer.py`), et le code ne doit pas dépendre de chemins Windows.
- **Disque** : environ 30 Go libres. Les checkpoints d'expériences occupent 72 Go dans `checkpoints/` ; on n'en supprime qu'avec l'accord de l'utilisateur.
- **Pur PyTorch**, sans extension C++/CUDA à compiler (règle existante du projet).
- **L'existant reste fonctionnel** : le pipeline quadrupède, ses 48 tests et le modèle publié sur Hugging Face (`MissawB/creature-flow-point-dit`) ne doivent pas casser. Le nouveau code vit à côté et réutilise ce qui se réutilise : flow matching, OT, EMA, entraîneur, métriques, visionneuse.
- **Licences** : Articulation-XL2.0 est sous CC-BY-4.0 (citer MagicArticulate). Les objets d'origine (Objaverse-XL, GitHub et Sketchfab) ont chacun leur propre licence. On publie le modèle, les métriques et la visionneuse ; on ne republie pas le jeu de données dérivé sans avoir vérifié les licences par objet.

## 4. Source des données (décidé)

**Articulation-XL2.0** (`Seed3D/Articulation-XL2.0` sur Hugging Face, CC-BY-4.0) : 48 637 modèles issus d'Objaverse-XL. Chacun fournit sommets, faces, normales, articulations, os, racine, noms d'articulations et poids de peau. Des catégories sont attribuées par modèle. La version Parquet `chibifire/datasource-articulation-xl2-parquet` permet de lire table par table (`skeletons`, `meshes`, `skins`, `meta`) sans télécharger le fichier de 31 Go ; elle annonce une vérification champ par champ contre l'original.

Répartition mesurée sur `meta_Articulation_XL_2.0.csv` :

| Catégorie (`category_label`) | Modèles |
|---|---|
| character | 14 126 |
| character, anthropomorphic | 9 166 |
| animal | 4 010 |
| mythical creature | 2 531 |
| anthropomorphic | 2 270 |
| mythical creature, anthropomorphic | 1 853 |
| anthropomorphic, toy | 981 |
| animal, anthropomorphic | 705 |
| … (puis armes, outils, véhicules, vêtements, plantes, architecture…) | |

Les squelettes vont de 2 à 100 os. Sur les animaux, il y a 25 articulations en médiane, 7 au 10e centile et 52 au 90e.

**Sélection (décidé par l'utilisateur)** : on garde tout ce qui a un corps de créature, y compris humanoïdes, personnages et jouets, soit environ 40 000 modèles. On écarte les objets sans corps : armes seules, outils, véhicules, vêtements, architecture, appareils, plantes, objets ménagers, « anatomy ». Le filtrage se fait par étiquettes de catégorie, puis par contrôles automatiques : modèle vide, un seul os, maillage dégénéré. **Équilibre (décidé par l'utilisateur)** : à l'entraînement, les tirages sont pondérés pour que **70 %** des exemples soient des animaux et des créatures mythiques, et **30 %** des humanoïdes, personnages et jouets, quelle que soit leur proportion brute (environ 7 000 contre 33 000). Les étiquettes mixtes du type « animal, anthropomorphic » comptent côté animaux. La règle exacte pour les étiquettes mixtes (par ex. « character, weapon ») est laissée au constructeur. Elle doit être consignée et réversible.

Des packs CC0 (Quaternius) peuvent servir d'appoint, mais ne sont pas nécessaires.

## 5. Les étapes

Les étapes sont séquentielles : chacune dépend de la précédente. Les étapes 1 et 2 sont spécifiées en détail. Les étapes 3 et 4 sont décrites dans leurs grandes lignes et seront précisées avec l'utilisateur quand on y arrivera.

### Étape 1 : données

**But** : transformer les modèles sélectionnés en exemples d'entraînement (squelette, nuage dense), normalisés et multipliés par déformation.

1. **Téléchargement partiel** des tables Parquet utiles. Les objets sélectionnés sont repérés par `uuid` via `meta`. On ne télécharge que ce qu'il faut.
2. **Échantillonnage de surface** : des points tirés uniformément selon l'aire des faces. Chaque point garde ses poids de peau, interpolés de façon barycentrique depuis les sommets. On stocke un nuage dense de référence par modèle (ordre de grandeur : 8192 à 16 384 points, à fixer selon le disque).
3. **Normalisation** : même échelle pour tous (par ex. sphère englobante unité ou hauteur unité), racine centrée, et **orientation canonique** (haut et avant cohérents d'un modèle à l'autre). **Convention (choix de conception)** : « haut » est la verticale de la pose d'origine (gravité, pieds au sol) et « avant » est la direction du regard. Chaque créature garde sa posture naturelle : un humanoïde reste debout, un quadrupède, un poisson ou un serpent reste à l'horizontale, tête en avant. C'est la convention dans laquelle les modèles 3D sont créés ; la posture d'un hybride bipède/quadrupède est une question de l'étape 3. L'orientation est le point délicat, car les fichiers Objaverse ne partagent ni axe vertical ni direction de face. Le constructeur choisit la méthode (heuristiques sur le squelette, axe Y des glTF, ACP…) et mesure le taux d'échec sur un échantillon inspecté visuellement.
4. **Augmentation par déformation du squelette** : les poids de peau permettent de déformer les points avec le squelette (skinning linéaire). On fait varier aléatoirement la longueur des os (répercutée sur les descendants), l'épaisseur autour des os, et la posture avec de petites rotations bornées. Chaque modèle donne ainsi de nombreuses variantes cohérentes. Ce rôle était tenu jusqu'ici par le générateur procédural. Le constructeur choisit entre une augmentation à la volée (calcul léger sur les points) et des variantes précalculées.
5. **Répartition** : on reprend la séparation train/test d'Articulation-XL. Tout objet évalué n'a été vu à l'entraînement sous aucune de ses déformations.
6. **Stockage compact** : demi-précision et 4 poids de peau principaux par point, pour tenir dans le budget disque (§3).

**Terminé quand** : un script reproductible produit le cache. Un rapport donne le nombre de modèles retenus par catégorie, le taux d'échec de normalisation et des planches de contrôle visuel (une vingtaine de créatures de catégories différentes, de face et de profil, avec leur squelette).

### Étape 2 : modèle à squelette libre, dense et rapide à entraîner

**But** : générer un nuage dense à partir de n'importe quel squelette, avec un visage lisible.

**Ce qui change par rapport au modèle actuel** (`models/point_dit.py`, `models/skeleton_encoder.py`) :

- **Squelette de taille variable.** L'encodeur actuel donne à chaque articulation une identité apprise (`nn.Embedding(35, …)`), ce qui suppose une topologie fixe. Il faut le remplacer par un encodage qui ne dépend que de la structure : position, position relative au parent, profondeur dans l'arbre, et un encodage du graphe (par ex. distances dans l'arbre ou vecteurs propres du laplacien). Les lots contiennent des squelettes de tailles différentes, complétés par du padding et un masque, dans l'encodeur comme dans la cross-attention points → articulations. Le biais de distance multi-échelle actuel ne dépend pas de la topologie et doit être conservé. Le constructeur peut l'étendre aux **os** (distance au segment plutôt qu'à l'articulation), plus adaptés aux os longs.
- **Un code de forme en plus du squelette (prévu dès cette étape).** Un squelette seul ne distingue pas toujours un bec d'un museau, ni un aigle d'un chien dont les os de la tête seraient semblables. Le modèle reçoit donc aussi un petit code latent $z$ par créature, produit par un encodeur de nuage de points réel. Il est entraîné avec le reste, avec un goulot d'étranglement : petite dimension, bruit ou régularisation, et abandon aléatoire du conditionnement. Il doit rester lisse à l'interpolation, et le modèle ne doit pas ignorer le squelette. C'est ce code qui permettra à l'étape 3 de mélanger les traits. **Origine de $z$ (choix de conception)** : pour une créature du catalogue, $z$ est encodé depuis son vrai nuage. Le code est aussi abandonné aléatoirement à l'entraînement (code nul), pour que le modèle sache générer **à partir du squelette seul**. Les métriques sur le jeu de test sont calculées dans les deux modes : squelette seul (mesure de génération) et squelette avec $z$ (mesure proche d'une reconstruction, rapportée comme telle). Le risque principal est que $z$ capte tout et que le squelette ne serve plus. L'évaluation doit vérifier que changer le squelette à $z$ fixé change bien la forme.
- **Plus de points, en deux temps (décidé par l'utilisateur : la vitesse d'abord, puis le détail du visage à la demande).**
  - *Passe globale* : un nuage de la créature entière, assez dense pour lire sa silhouette et sa tête, mais dont la génération doit prendre **1 à 2 s** dans la visionneuse. Le nombre de points est fixé par ce budget de temps : environ 2048 à 4096, à mesurer.
  - *Passe « visage »* : à la demande, un nuage dense **limité à la tête** (par ex. plusieurs milliers de points dans le rayon des articulations de la tête), qui se superpose à la passe globale et peut prendre plus de temps. Deux pistes laissées au constructeur : un modèle de détail entraîné sur des recadrages de la tête dans un repère local (conditionné par le sous-arbre de la tête et le code $z$), ou le même modèle interrogé avec des points restreints à la région de la tête. Le visage détaillé doit rester raccord avec la passe globale (même position, même échelle, pas de couture visible au cou). Pour la passe globale comme pour la passe visage, le constructeur peut, pour garder un entraînement rapide, entraîner sur des sous-ensembles tirés au hasard du nuage dense et générer plus dense, à condition de vérifier que la qualité tient quand on change le nombre de points. Sinon, il entraîne directement au nombre cible, avec l'attention efficace de PyTorch (`scaled_dot_product_attention`) en bf16.
- **Couplage OT.** Sur les quadrupèdes, le couplage OT par forme a été le plus gros gain mesuré (écart médian à la surface 0.0052 → 0.0040, loss 1.29 → 0.013). Mais l'appariement exact précalculé (environ 2 s par tirage et par forme sur CPU) ne passe pas à 40 000 modèles et plus de points. Le constructeur doit le remplacer par un appariement **calculé sur le GPU pendant l'entraînement** (par ex. Sinkhorn suivi d'une affectation, ou une approximation par OT tranché). Il faut d'abord valider sur l'ancien jeu quadrupède qu'il redonne des résultats proches de l'appariement exact : écart médian proche de 0.0040 à 90 époques, contre 0.0052 sans OT. Nous avons mesuré que 2 tirages précalculés valent 8 ; un appariement en ligne en fournit autant qu'on veut.
- **Entraînement rapide.** Les choix existants restent : bf16, EMA 0.995, `data_scale`, Euler 10 en génération. Les données lourdes sont précalculées une fois. Le modèle reste compact, de l'ordre de 10 à 30 M de paramètres, pour l'entraînement comme pour l'inférence dans le navigateur. Les runs longs se font sur Kaggle ou Colab avec reprise sur checkpoint. Viser d'abord un premier modèle utile en **quelques heures** sur la RTX 4060, sur un sous-ensemble si nécessaire.
- **Préparer le mélange dès cette étape (choix de conception)**, pour ne pas réentraîner à l'étape 3. Chaque articulation reçoit une valeur de présence ou de poids en entrée (1 pour un squelette réel). L'augmentation rétrécit ou retire aléatoirement des sous-arbres, en abaissant cette valeur, et le nuage cible est déformé en conséquence par skinning, ou les points de la partie retirée sont exclus. L'encodeur de forme est régularisé pour que les combinaisons linéaires de codes restent dans la distribution apprise.
- **Seuil de validation de l'OT en ligne (choix de conception)** : réussi si l'écart médian à la surface est ≤ 0.0045 sur le jeu quadrupède à 90 époques (exact : 0.0040, sans OT : 0.0052). Sinon : une passe de réglage (itérations, régularisation), puis repli sur un appariement exact précalculé pour un sous-ensemble de points et de formes. Le résultat est consigné dans tous les cas.

**Évaluation**
- Sur le test d'Articulation-XL : MMD-CD, COV-CD et 1-NNA-CD, avec le détail côté générées / côté réelles déjà affiché par `scripts/evaluate_generation.py`.
- L'**écart à la surface** se mesure désormais par rapport au vrai maillage (distance point → maillage), à la place des cylindres d'os.
- Une **planche fixe** d'une douzaine de créatures de test (mammifère, oiseau, reptile ou dragon, poisson ou serpent, insecte, humanoïde, jouet) avec gros plans de la tête : c'est le critère « visage lisible », jugé à l'œil par l'utilisateur.
- Le temps d'entraînement et le temps de génération sont rapportés avec chaque résultat.

**Terminé quand** : sur la planche de test, l'utilisateur reconnaît la tête, le visage et les appendices des créatures jamais vues, et les métriques sont consignées dans le README comme pour l'ancien pipeline.

### Étape 3 : mélange pondéré (grandes lignes)

**But** : un enfant cohérent à partir de deux ou trois parents et de leurs pourcentages.

Direction retenue (choix de conception, à confirmer quand on y sera) :
- **Squelette de l'enfant** : on met les squelettes des parents en correspondance, articulation par articulation, en normalisant leur position et leur rôle dans l'arbre (par ex. affectation optimale). Les parties appariées sont interpolées selon les pourcentages. Les parties propres à un seul parent (ailes, deuxième tête, queue) apparaissent, grandissent ou s'effacent avec le poids de ce parent ; leur règle exacte reste à concevoir.
- **Traits de l'enfant** : son code de forme est la combinaison pondérée des codes des parents, $z = \sum_i w_i z_i$.
- Le modèle fait pousser une chair continue sur ce squelette avec ce code. Comme il n'a appris que des créatures cohérentes, c'est lui qui doit éviter le patchwork.

**Mode aléatoire** : parents tirés dans le catalogue (de préférence de catégories différentes), poids tirés au hasard (par ex. Dirichlet). La recette tirée (parents et pourcentages) est affichée pour être rejouée ou retouchée (idée proposée, pas encore validée par l'utilisateur).

**Réussi quand** : faire glisser un curseur de 0 à 100 % transforme l'enfant continûment d'un parent à l'autre, et un mélange 50/50 entre deux créatures très différentes est jugé cohérent par l'utilisateur.

### Étape 4 : visionneuse (grandes lignes)

- **Catalogue** de toutes les créatures, avec recherche, choix de deux ou trois parents, curseur de pourcentage par parent, et **bouton aléatoire**.
- **Génération en direct**, puisque les combinaisons ne se précalculent pas. Choix par défaut : faire tourner le modèle **dans le navigateur** (ONNX Runtime Web, WebGPU), ce qui reste gratuit et compatible avec le Space statique actuel `MissawB/creature-flow-viewer`. Repli si c'est trop lent : un Space Gradio sur CPU gratuit.
- Un bouton **« Détailler le visage »** lance la passe visage et rapproche la caméra de la tête.
- L'animation du flux $t = 0 \to 1$ et l'affichage du squelette restent ceux de la visionneuse actuelle (`viewer/app.js`, Three.js).
- **Points ouverts** :
  - Articulation-XL n'a pas de noms lisibles. Il faudra en tirer de `fileIdentifier`, des annotations Objaverse ou de légendes existantes, et les vignettes devront être rendues depuis nos nuages.
  - Embarquer environ 40 000 squelettes et codes dans la page peut peser des dizaines de Mo ; un catalogue choisi plus restreint est possible.

## 6. Ce qui est laissé au constructeur

- La façon de produire la passe visage (modèle de détail dédié ou même modèle restreint à la tête).
- La règle de filtrage des catégories mixtes, la méthode d'orientation canonique, le nombre de points stockés.
- Le choix entre augmentation à la volée et variantes précalculées.
- L'encodage du graphe du squelette, la forme du biais (articulations ou os), la taille du modèle.
- La méthode d'OT en ligne, à condition de la valider contre l'appariement exact.
- Entraîner sur des sous-ensembles de points ou directement au nombre cible.
- L'architecture de l'encodeur de forme et sa régularisation.

Chaque choix est consigné, avec la mesure qui l'a justifié quand il y en a une, comme dans le README actuel.

## 7. Risques connus

- **Qualité hétérogène** des modèles amateurs : géométrie interne (yeux, dents, intérieur de bouche), pièces détachées, vêtements, armes tenues en main. Ces éléments peuvent être gardés, mais il faut les surveiller dans les planches.
- **Orientation et échelle** incohérentes entre fichiers (§4, étape 1.3).
- **Le code de forme qui court-circuite le squelette** (§5, étape 2).
- **Coût d'inférence dans le navigateur** : la passe globale doit tenir en 1 à 2 s.
- **Disque** : environ 30 Go libres.

## 8. Références

- Song et al., *MagicArticulate: Make Your 3D Models Articulation-Ready*, CVPR 2025 (Articulation-XL2.0).
- Gat et al., *AnyTop: Character Animation Diffusion with Any Topology*, 2025 (encodage de squelettes de topologie quelconque, correspondances sémantiques entre articulations).
- Zhang et al., *One Model to Rig Them All: Diverse Skeleton Rigging with UniRig*, 2025 (pose automatique de squelette, piste pour élargir les données plus tard).
- Tong et al., *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport*, 2023 (OT en ligne).
