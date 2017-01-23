# Regresja logistyczna - wstęp
import matplotlib.pyplot as plt
import numpy as np


def transformation(x):
    #return 0.1 * x + 0.02
    a = -2.4285
    b = -(a) * 9
    return a*x + b


X = [1, 2, 4, 7, 8, 10, 12, 17]  # rozmiary guzów
Y = [0, 0, 0, 0, 0, 1, 1, 1]  # flaga oceniająca ich złośliwość - 1=złośliwy / 0=niezłośliwy
function_sampling_x_coordinates = np.linspace(0, 18, 500)  # generowanie 500 punktów pomiędzy granicami osi X (0,18)
function_sampling_y_coordinates = [transformation(x) for x in
                                   function_sampling_x_coordinates]  # wyznaczanie wartosci funkcji transformation w tych punktach

# ---Zadanie 1---
# Zamień postać funkcji transformation tak, aby stworzyła krzywą logistyczną
# Jaki wpływ ma zmiana wartości parametru a na funkcję?
    ## wpływa na skos funkcji
# Jaki wpływ ma zmiana wartości parametru b na funkcję?
    ## miejsce, w którym prosta przetnie się z osią OX

plt.plot(X, Y, "o")
plt.plot(function_sampling_x_coordinates, function_sampling_y_coordinates, 'r-')
plt.ylim(ymax=1.25, ymin=-0.25)

plt.show()