# IA_TP1

**Question 1**
*Quelle sera la prochaine valeur du poids du neurone gagnant dans le cas où η = 0 ? η = 1 ? En déduire
 géométriquement la prochaine valeur du poids dans le cas normal où η ∈]0, 1[.*  
  Si η = 0, comme la différence du poids calculé et du poids précédent, qui est définit par le produit du taux d'apprentissage, de la fonction gaussienne de voisinage  et de (xi - wij),
  la différence sera de 0. Si la différence est nul, cela signifie que le poids actuel est égal au poids précédent.
  
  η = 1, la prochaine valeur du poids sera proportionnel à sa distance, le poids sera donc constant.  
  Plus le taux d'apprentissage sera proche de 0, moins les poids changeront. Plus le taux d'apprentissage sera proche de 1,
   j sera égal à j* donc la diff de poids ser la diff entre l'entré et le poid donc la nouvelle valeur du poids sera la valeur d'entrée
  
*— Si σ augmente, est-ce que les neurones vont plus ou moins apprendre l’entrée courante ? En déduire l’influence
 que doit avoir σ sur la “grille” de neurone, sera-t-elle plus “lâche” ou plus “serrée” si σ augmente ?*
   
 Plus σ augmente, plus la fonction de voisinage s'approchera de 1.  
 Plus σ augmente, plus la grille de neurone sera lâche. Plus σ diminue, plus la grille de neurone serrée.  
 Plus la grille de neurone est serrée, plus les neurones vont apprendre.

*— Prenons le cas d’une carte avec un seul neurone qui reçoit 2 entrées x 1 et x 2 . Durant l’apprentissage x 1 (res-
 pectivement x 2 ) est présenté n 1 (respectivement n 2 ) fois. Après l’apprentissage où se situera géométriquement
 le poids du neurone ?*  
 S'il n'y a qu'un seul neurone, j* sera égal à j. L'exposant de la fonction exponentielle sera alors égal à 0, on aura donc comme différence de poids :  
 η * (xi - wij)
 On aura donc  
 η * (x1 - w11) * n1  
 η * (x2 - w21) * n2  
 
   

**Question 2**  
Distance euclidienne entre deux points du plan x et y avec les fonctions de numpy
```python
numpy.sqrt(numpy.sum(numpy.power(x - y,2)))
```
On l'implémente dans la fonction compute pour calculer la distance entre le poids et la nouvelle entrée

Ensuite on implémente la fonction pour calculer le poids dans la fonction learn
```python
distCarre = (self.posx - posxbmu) ** 2 + (self.posy - posybmu) ** 2
ftnVoisinage = numpy.exp(-distCarre / (2 * sigma **2))
self.weights[:] = eta * ftnVoisinage * (x - self.weights) + self.weights
```
**Question 3**  
Vartiation ETA
    ETA = 0.01 variatio plus lente ?

**Question 4**  
