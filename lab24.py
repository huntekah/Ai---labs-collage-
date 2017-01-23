import itertools
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

digits = load_digits()  # zbiór zawiera 1797 obrazków reprezentujących cyfry od 0 do 9

fig = plt.figure(figsize=(4, 4))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(16):
    # wyświetl próbkę 16 obrazków wraz z ich prawdziwymi etykietami
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    ax.text(0, 7, str(digits.target[i]))
plt.show()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=16)

# --- Zadanie 4 ---
# Na podstawie dokumentacji sklearn, stwórz MLPClassifier, wytrenuj go a następnie użyj wytrenowanego modelu do
# przewidywania etykiet na zbiorze testowym
# zinstancjonuj klasyfikator w zmiennej mlp
# listę przewidzianych etykiet umieść w obiekcie predicted
mlp = MLPClassifier()
mlp.fit(X_train, y_train)
predicted = mlp.predict(X_test)

assert predicted != None
# Wizualizacja macierzy pomyłek - pokazuje jakie etykiety pomylono z jakimi innymi
cm = confusion_matrix(y_test, predicted)
print(cm)

expected = y_test
fig = plt.figure(figsize=(8, 8))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# Wizualizacja błędnie zaanotowanych obrazków
cnt = 0
for i in range(len(X_test)):
    if predicted[i] == expected[i]:
        continue
    ax = fig.add_subplot(8, 8, cnt + 1, xticks=[], yticks=[])
    ax.imshow(X_test.reshape(-1, 8, 8)[i], cmap=plt.cm.binary, interpolation='nearest')
    cnt += 1

    ax.text(0, 7, "generated:" + str(predicted[i]), color='red')
    ax.text(0, 6, "expected:" + str(expected[i]), color='green')
plt.show()