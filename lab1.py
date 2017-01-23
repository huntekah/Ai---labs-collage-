import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_regression

sizes = np.loadtxt('flat.areas').reshape((26, 1))  # załaduj wektor rozmiarów mieszkań
prices = np.loadtxt('flat.prices').reshape((26, 1))  # załaduj wektor cen mieszkań
samples_cnt = len(prices)  # pobierz liczność par rozmiar->cena
weight0 = 0.0  # inicjalizuj wagi
weight1 = 0.0  #
learning_rate = 0.0001  # stala uczenia
maxIteration = 100000  # liczba iteracji


def predict_prices(w0, w1, _list):
    ret = [w0 * i[0] + w1 for i in _list]
    #print(ret)
    return ret


def refreshed_weights(a, b, predicted_prices):
    m = len(sizes)
    sumaA = sum( [(predicted_prices[i] - prices[i])*sizes[i] for i in range(m) ] )
    sumaB = sum( [(predicted_prices[i] - prices[i]) for i in range(m) ] )
    a = a - learning_rate * (1.0 / m)* sumaA
    b = b - learning_rate * (1. / m) * sumaB
    return a,b

for i in range(maxIteration):
    weight0 = weight0
    weight1 = weight1

    # Zadanie1:
    predicted_prices = []
    #print(sizes)
    # uzupełnij listę predicted_prices tak, aby dla każdego metrazu budynku z listy 'sizes' wyznaczyć cenę tegoż
    # budynku przy użyciu aktualnych wag modelu liniowego
    predicted_prices = predict_prices(weight0, weight1, sizes)
    #print(predicted_prices)


    # Zadanie2:
    # napisz kod, który iteracyjnie poprawiać będzie wagi tak, aby ostatecznie wyznaczyły prostą,
    # która najlepiej odwzoru je zależność metraż -> cena
    weight0, weight1 = refreshed_weights(weight0, weight1, predicted_prices)

plt.plot(sizes, prices, "x")
if len(predicted_prices) > 0:
    plt.plot(sizes, predicted_prices, "r-")
plt.title('Ceny mieszkan w zaleznosci od metrazu')
plt.xlabel('metraz (m^2)')
plt.ylabel('Cena mieskania (tys. zł)')
plt.show()
print("Wyznaczone wartości wag -> w0:", weight0, "w1:", weight1)