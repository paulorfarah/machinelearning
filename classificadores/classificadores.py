#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
from datetime import datetime

import matplotlib
from sklearn.datasets import load_svmlight_file
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import scikitplot as skplt


def knn(x_train, x_test, y_train, y_test, k, metric, weight):

    # cria um kNN
    neigh = KNeighborsClassifier(n_neighbors=int(k), metric=metric, weights=weight)

    # print 'Fitting knn'
    neigh.fit(x_train, y_train)

    # mostra o resultado do classificador na base de teste
    acuracia = neigh.score(x_test, y_test)
    # print acuracia

    # predicao do classificador
    # print 'Predicting...'
    # y_pred = neigh.predict(x_test)
    # # cria a matriz de confusao
    # cm = confusion_matrix(y_test, y_pred)
    # print cm

    # skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=y_pred, normalize=True,
    #                                     title="Matrix de Confusao")
    # plt.tight_layout()
    # plt.savefig("confusion_matrix_knn.pdf")
    # plt.show()

    return acuracia

def naive_bayes(X_train, x_test, y_train, y_test):
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(x_test)
    # print("Number of mislabeled points out of a total %d points : %d" % (X_train.shape[0], (y_train != y_pred)))
    #return "%d" % ((np.array(y_train) != y_pred).sum()) #(y_train != y_pred)
    acuracia = gnb.score(x_test, y_test)

    # y_pred = gnb.predict(x_test)
    # # cria a matriz de confusao
    # cm = confusion_matrix(y_test, y_pred)
    # print cm
    # skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=y_pred, normalize=True,
    #                                     title="Matrix de Confusao")
    # plt.tight_layout()
    # plt.savefig("confusion_matrix_nb.pdf")
    # plt.show()

    return acuracia

def lda(X_train, X_test, y_train, y_test):
    # print 'LDA'
    #fonte: https://stackabuse.com/implementing-lda-in-python-with-scikit-learn/
    #Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    lda = LDA(n_components=1)
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)

    lda.fit(X_train, y_train)

    y_pred = lda.predict(X_test)

    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)

    acuracia = accuracy_score(y_test, y_pred)
    # print('Accuracy' + str(accuracy_score(y_test, y_pred)))

    # # cria a matriz de confusao
    # cm = confusion_matrix(y_test, y_pred)
    # print cm
    #
    # skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=y_pred, normalize=True,
    #                                     title="Matrix de Confusao")
    # plt.tight_layout()
    # plt.savefig("confusion_matrix_lda.pdf")
    # plt.show()

    return acuracia

def logistic_regression(X_train, X_test, y_train, y_test):
    # print 'Logistic Regression'
    #fonte: https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
    # all parameters not specified are set to their defaults
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train, y_train)

    # Returns a NumPy Array
    # Predict for One Observation (image)
    # logisticRegr.predict(X_test[0].reshape(1, -1))

    # Predict for Multiple Observations(images) at Once
    # logisticRegr.predict(X_test[0:10])

    #Make predictions on entire test data
    y_pred = logisticRegr.predict(X_test)

    # Use score method to get accuracy of model
    acuracia = logisticRegr.score(X_test, y_test)
    # print(acuracia)

    # cm = metrics.confusion_matrix(y_test, y_pred)
    # print(cm)
    #
    # skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=y_pred, normalize=True,
    #                                     title="Matrix de Confusao")
    # plt.tight_layout()
    # plt.savefig("confusion_matrix_lr.pdf")
    # plt.show()

    return acuracia

def main(arq_treino, arq_teste):
    print '### Impactos da base de Aprendizagem ###'

    resultado = []
    # ler dataset de treino
    print 'lendo datasets, aguarde...' + str(arq_treino)
    X_data, y_data = load_svmlight_file(arq_treino)
    acuracias_knn = []
    acuracias_nb = []
    acuracias_lda = []
    acuracias_lr = []

    tempos_knn = []
    tempos_nb = []
    tempos_lda = []
    tempos_lr = []
    for aux_size in range(4, 104, 4):
        train_size = float(aux_size)/100.0
        #ler dataset de treino
        try:
            if train_size >= 1:
                X_train, _, y_train, _ = train_test_split(X_data, y_data, test_size=0, random_state=5)
            else:
                X_train, _, y_train, _ = train_test_split(X_data, y_data, train_size=train_size, test_size=0, random_state=5)
            tam_treino = X_train.shape[0]

            #ler dataset de teste

            X_test, y_test = load_svmlight_file(arq_teste)

            X_train = X_train.toarray()
            X_test = X_test.toarray()

            a = datetime.now()
            res_knn = knn(X_train, X_test, y_train, y_test, 9, 'euclidean', 'distance')
            b = datetime.now()
            tempos_knn.append(b-a)
            a = datetime.now()
            res_nb = naive_bayes(X_train, X_test, y_train, y_test)
            b = datetime.now()
            tempos_nb.append(b-a)
            a = datetime.now()
            res_lda = lda(X_train, X_test, y_train, y_test)
            b = datetime.now()
            tempos_lda.append(b-a)
            a = datetime.now()
            res_lr = logistic_regression(X_train, X_test, y_train, y_test)
            b = datetime.now()
            tempos_lr.append(b-a)
            # resultado.append([tam_treino, [res_knn, res_nb, res_lda, res_lr]])
            print str(tam_treino) + '; ' + str(res_knn) + '; ' + str(res_nb) + '; ' + str(res_lda) + '; ' + str(res_lr)
            acuracias_knn.append(res_knn)
            acuracias_nb.append(res_nb)
            acuracias_lda.append(res_lda)
            acuracias_lr.append(res_lr)

        except:
            print 'Erro: ' + str(sys.exc_info())

    # plotar acuracia
    ind = np.arange(25)  # the x locations for the groups
    # width = 0.6  # the width of the bars
    plt.plot(acuracias_knn, marker='P', label='kNN')
    plt.plot(acuracias_nb, marker='D', label='Naive Bayes')
    plt.plot(acuracias_lda, marker='*', label='LDA')
    plt.plot(acuracias_lr, marker='v', label='Logistic Regression')
    plt.legend()
    plt.xticks(ind, (
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
    '17', '18', '19', '20', '21', '22', '23', '24', '25'))

    #save
    plt.savefig('acuracia.pdf')
    plt.close()
    print tempos_knn
    print tempos_nb
    print tempos_lda
    print tempos_lr 

    import csv

    # abrindo o arquivo para escrita
    with open('acuracia.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(acuracias_knn)
        spamwriter.writerow(acuracias_nb)
        spamwriter.writerow(acuracias_lda)
        spamwriter.writerow(acuracias_lr)

    # abrindo o arquivo para escrita
    with open('tempos.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(tempos_knn)
        spamwriter.writerow(tempos_nb)
        spamwriter.writerow(tempos_lda)
        spamwriter.writerow(tempos_lr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print 'exemplo de uso: python classificadores.py train test (' + str(len(sys.argv)) + ')'
    else:
        arq_treino = sys.argv[1]
        arq_teste = sys.argv[2]
        main(arq_treino, arq_teste)