import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn.model_selection import cross_val_score

def plot_boundary(elect_train, dt1, dt2 ):
    y_train = elect_train['won']
    x1_min, x1_max = elect_train['minority'].min() - 1, elect_train['minority'].max() + 1
    x2_min, x2_max = elect_train['bachelor'].min() - 1, elect_train['bachelor'].max() + 1
    x1x, x2x = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                         np.arange(x2_min, x2_max, 0.1))

    yhat2 = dt1.predict(np.c_[x1x.ravel(), x2x.ravel()]).reshape(x1x.shape)
    yhat10 = dt2.predict(np.c_[x1x.ravel(), x2x.ravel()]).reshape(x1x.shape)


    fig, (ax1,ax2) = plt.subplots(1,2, figsize=[15,6])

    ax1.contourf(x1x, x2x, yhat2, alpha=0.2,cmap="PiYG");
    ax1.scatter(elect_train['minority'][y_train==1], elect_train['bachelor'][y_train==1],marker=".",color="green",label="Trump")
    ax1.scatter(elect_train['minority'][y_train==0], elect_train['bachelor'][y_train==0],marker=".",color="purple",label="Clinton")

    ax1.set_xlabel("minority")
    ax1.set_ylabel("bachelor")
    ax1.set_title("Decision Tree with max_depth=2")
    ax1.legend()

    ax2.contourf(x1x, x2x, yhat10, alpha=0.2,cmap="PiYG");
    ax2.scatter(elect_train['minority'][y_train==1], elect_train['bachelor'][y_train==1],marker=".",color="green",label="Trump")
    ax2.scatter(elect_train['minority'][y_train==0], elect_train['bachelor'][y_train==0],marker=".",color="purple",label="Clinton")

    ax2.set_xlabel("minority")
    ax2.set_ylabel("bachelor")
    ax2.set_title("Decision Tree with max_depth=10")
    ax2.legend()
    plt.show()

def plot_boundary_2(elect_train, dt1, dt2, feature_columns, labels):
    """
    Función para graficar los límites de decisión de dos modelos de árbol de decisión.

    Args:
        elect_train (DataFrame): Conjunto de datos de entrenamiento.
        dt1 (DecisionTreeClassifier): Primer modelo de árbol de decisión.
        dt2 (DecisionTreeClassifier): Segundo modelo de árbol de decisión.
        feature_columns (list): Lista con los nombres de las dos columnas a usar como características.
        labels (list): Lista con los nombres de las etiquetas para los ejes [x_label, y_label].
    """
    # Extraer las características y el objetivo
    x1, x2 = feature_columns
    y_train = elect_train['won']

    # Definir los límites del grid
    x1_min, x1_max = elect_train[x1].min() - 1, elect_train[x1].max() + 1
    x2_min, x2_max = elect_train[x2].min() - 1, elect_train[x2].max() + 1
    x1x, x2x = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                           np.arange(x2_min, x2_max, 0.1))

    # Realizar predicciones con ambos modelos
    X_input = np.c_[x1x.ravel(), x2x.ravel()]
    yhat2 = dt1.predict(X_input).reshape(x1x.shape)
    yhat10 = dt2.predict(X_input).reshape(x1x.shape)

    # Crear las gráficas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[15, 6])

    # Gráfica para el primer modelo
    ax1.contourf(x1x, x2x, yhat2, alpha=0.2, cmap="PiYG")
    ax1.scatter(elect_train[x1][y_train == 1], elect_train[x2][y_train == 1], marker=".", color="green", label="Trump")
    ax1.scatter(elect_train[x1][y_train == 0], elect_train[x2][y_train == 0], marker=".", color="purple", label="Clinton")
    ax1.set_xlabel(labels[0])
    ax1.set_ylabel(labels[1])
    ax1.set_title("Decision Tree with max_depth=2")
    ax1.legend()

    # Gráfica para el segundo modelo
    ax2.contourf(x1x, x2x, yhat10, alpha=0.2, cmap="PiYG")
    ax2.scatter(elect_train[x1][y_train == 1], elect_train[x2][y_train == 1], marker=".", color="green", label="Trump")
    ax2.scatter(elect_train[x1][y_train == 0], elect_train[x2][y_train == 0], marker=".", color="purple", label="Clinton")
    ax2.set_xlabel(labels[0])
    ax2.set_ylabel(labels[1])
    ax2.set_title("Decision Tree with max_depth=10")
    ax2.legend()

    plt.show()