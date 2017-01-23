import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from helpers_lab2 import plot_boundary

iris = datasets.load_iris()
X = iris.data[:, :2]  #weź tylko dwie cechy przykładowego zbioru
Y = iris.target # Y to klasy, które chcemy przewidzieć

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X, Y)
#print(X)
#print(Y)

plot_boundary(logreg, X, Y)
#Zadanie 2:
#Na podstawie dokumentacji sklearn, stwórz klasyfikator LogisticRegression i wytrenuj go odpowiednio, aby dobrze separował przestrzeń.