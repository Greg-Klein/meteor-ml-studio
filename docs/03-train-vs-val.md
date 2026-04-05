# 3. Comprendre `train` et `val` 🧪

## Pourquoi il y a deux ensembles

Quand on entraine un modele, il ne faut pas tout melanger.

On separe les donnees en au moins deux groupes :

- `train`
- `val`

## `train` 📘

Le dossier `train` contient les images qui servent a apprendre.

Le modele voit ces images pendant l'entrainement.

## `val` 🔍

Le dossier `val` contient les images qui servent a verifier si le modele generalise bien.

Le modele ne doit pas apprendre dessus.

## Exemple simple

Tu as 100 images.

Tu peux faire :

- 80 images dans `train`
- 20 images dans `val`

Pendant l'entrainement :

- YOLO apprend sur les 80 images de `train`
- YOLO mesure ses performances sur les 20 images de `val`

## Pourquoi c'est important

Si tu evalues le modele sur les images qu'il a deja vues pour apprendre, tu risques de te tromper.

Le modele peut :

- memoriser
- avoir l'air tres bon
- etre en realite mauvais sur de nouvelles images

## Dans ton cas

Le mieux est de separer les images par nuits, pas juste au hasard.

Par exemple :

- nuits 1 a 8 dans `train`
- nuits 9 et 10 dans `val`

Pourquoi :

- des images d'une meme nuit se ressemblent souvent beaucoup
- sinon l'evaluation serait trop optimiste

## Plus tard : `test`

On peut aussi ajouter un troisieme ensemble :

- `test`

Mais pour commencer :

- `train`
- `val`

cela suffit.

## ✅ A retenir

- `train` = apprendre
- `val` = verifier
- il ne faut pas les melanger
