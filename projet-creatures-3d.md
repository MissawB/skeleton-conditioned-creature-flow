# Génération compositionnelle de créatures 3D par flow matching conditionné sur squelette

## 1. Résumé

Ce projet vise à générer des **créatures 3D originales** et anatomiquement plausibles, représentées par des **nuages de points**, en s'appuyant sur les modèles génératifs récents (diffusion, score matching, flow matching). La génération est **conditionnée par une structure squelettique** (arbre de joints) et peut **composer des parties issues d'espèces différentes** tout en garantissant la cohérence géométrique aux jonctions.

## 2. Problématique

> Comment générer des créatures 3D originales, anatomiquement plausibles, en combinant diffusion / flow matching sur nuages de points et conditionnement par une structure squelettique, de façon à composer des parties issues d'espèces différentes tout en garantissant la cohérence aux jonctions ?

**Version courte :** *Génération compositionnelle de créatures 3D par flow matching conditionné sur squelette.*

## 3. Motivation

- La génération de nuages de points 3D par modèles de diffusion / flow matching est un domaine de recherche actif (Point-E, LION, PVD, approches en flow matching).
- Le conditionnement par squelette apporte un **contrôle sémantique** : le squelette (et le type d'animal qu'on en déduit) guide la forme générée.
- La composition de parties (tête, corps, membres, queue) ouvre la voie à des **créatures hybrides** réellement nouvelles, au-delà de l'interpolation entre exemples connus.

## 4. Questions de recherche

1. **Représentation** : comment encoder un squelette (arbre de joints, type d'animal) pour conditionner efficacement un modèle sur nuages de points ?
2. **Composition** : comment générer ou assembler des parties venant d'espèces différentes avec une continuité géométrique aux interfaces ?
3. **Originalité** : comment mesurer qu'une créature est nouvelle (et non une copie du jeu d'entraînement) tout en restant plausible ?

## 5. Hypothèse de travail

Un conditionnement par graphe squelettique, combiné à une génération par parties avec contrainte de cohérence aux jonctions, produit des créatures **plus plausibles et plus diversifiées** qu'une génération globale non structurée.

## 6. Approche

### Étape 1 — Baseline
Diffusion / flow matching sur nuages de points d'une catégorie restreinte (ex. quadrupèdes). Objectif : une baseline reproductible.

### Étape 2 — Conditionnement par squelette
- Représentation du squelette sous forme d'arbre / graphe.
- Encodeur de graphe (GNN ou transformer sur graphe) produisant un vecteur de conditionnement.
- Extension possible : algorithme prenant en entrée le squelette et **inférant le type d'animal** (arbre de décision ou classifieur).

### Étape 3 — Composition et jonctions
- Génération par parties (tête, corps, membres, queue).
- Assemblage avec contrainte de cohérence aux interfaces (continuité des surfaces, alignement des points de raccord).
- Génération de créatures hybrides par composition de squelettes.

### Étape 4 — Rendu 2D (bonus)
Création d'images 2D des créatures à partir du rendu du nuage de points / maillage, avec un modèle pré-entraîné (ex. génération d'images conditionnée de type ControlNet). Cette partie est une **démonstration finale**, pas une contribution scientifique.

## 7. Jeux de données

Le principal point de vigilance du projet : les jeux de données d'animaux 3D avec squelettes sont rares ou hétérogènes.

| Source | Intérêt | Limite |
|---|---|---|
| ShapeNet | Large, standard pour les nuages de points | Peu d'animaux |
| Objaverse (filtré) | Très grande diversité | Qualité hétérogène, squelettes absents |
| SMAL | Modèle paramétrique de quadrupèdes | Catégories limitées |
| DeformingThings4D | Animaux animés | Taille et variété limitées |
| AnimalPose | Poses d'animaux | 2D principalement |

**Action prioritaire :** vérifier la disponibilité et la qualité des données avant de figer le sujet.

## 8. Évaluation

- **Qualité géométrique :** Chamfer Distance, EMD, couverture (COV) et MMD.
- **Plausibilité anatomique :** cohérence squelette ↔ géométrie, proportions.
- **Originalité :** distance au plus proche voisin dans le jeu d'entraînement.
- **Comparaison :** baseline sans conditionnement squelette.

## 9. Risques et points de vigilance

- **Données** : corpus d'animaux 3D avec squelette limité.
- **Définition de la « jonction »** : à formaliser précisément (fusion de parties, raccord de parties générées séparément).
- **Définition de l'« originalité »** : nécessite une métrique claire pour ne pas confondre nouveauté et bruit.
- **Périmètre** : la partie images 2D réalistes est un projet à part entière, à traiter comme un bonus.

## 10. Périmètre minimal (si le temps est limité)

Se limiter au **conditionnement par squelette** (étapes 1 et 2) sans la composition de parties, avec une évaluation claire face à la baseline non conditionnée.

## 11. Références de départ

- Point-E (OpenAI), LION, PVD : génération de nuages de points par diffusion.
- Flow matching et rectified flow : formulations récentes pour la génération.
- SMAL : modèle paramétrique d'animaux quadrupèdes.
