#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import csv
from datetime import datetime

import numpy
# from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import load_svmlight_file
# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import scikitplot as skplt

def salvar_arquivo(arquivo, acuracia, tempo_busca, tempo_treino, tempo_total, vetores):
    # abrindo o arquivo para escrita
    with open(arquivo, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(acuracia)
        spamwriter.writerow(tempo_busca)
        spamwriter.writerow(tempo_treino)
        spamwriter.writerow(tempo_total)
        spamwriter.writerow(vetores)


def knn(X_train, x_test, y_train, y_test, k, metric, weight):
    # cria um kNN
    neigh = KNeighborsClassifier(n_neighbors=int(k), metric=metric, weights=weight)

    # print 'Fitting knn'
    t1 = datetime.now()
    neigh.fit(X_train, y_train)
    t2 = datetime.now()
    # mostra o resultado do classificador na base de teste
    acuracia = neigh.score(x_test, y_test)
    # print acuracia

    # predicao do classificador
    # print 'Predicting...'
    y_pred = neigh.predict(x_test)
    # # cria a matriz de confusao
    # cm = confusion_matrix(y_test, y_pred)
    # print cm

    skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=y_pred, normalize=True,
                                        title="Matrix de Confusao")
    plt.tight_layout()
    plt.savefig("resultados/confusion_matrix_knn.pdf")
    # plt.show()

    # #scatter plot
    # plt.subplot(211)
    # plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train, s=25, edgecolor='k')
    # # plt.scatter(neigh.support_vectors_[:, 0], neigh.support_vectors_[:, 1], marker='x')
    # plt.savefig("scatter_knn.pdf")

    return [acuracia, t2-t1]

def naive_bayes(X_train, x_test, y_train, y_test):
    gnb = GaussianNB()
    t1 = datetime.now()
    y_pred = gnb.fit(X_train, y_train).predict(x_test)
    t2 = datetime.now()
    # print("Number of mislabeled points out of a total %d points : %d" % (X_train.shape[0], (y_train != y_pred)))
    #return "%d" % ((np.array(y_train) != y_pred).sum()) #(y_train != y_pred)
    acuracia = gnb.score(x_test, y_test)

    y_pred = gnb.predict(x_test)
    # # cria a matriz de confusao
    # cm = confusion_matrix(y_test, y_pred)
    # print cm
    skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=y_pred, normalize=True,
                                        title="Matrix de Confusao")
    plt.tight_layout()
    plt.savefig("resultados/confusion_matrix_nb.pdf")
    # plt.show()

    # scatter plot
    # plt.subplot(211)
    # plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train, s=25, edgecolor='k')
    # # plt.scatter(gnb.support_vectors_[:, 0], gnb.support_vectors_[:, 1], marker='x')
    # plt.savefig("scatter_nb.pdf")

    return [acuracia, t2-t1]

def lda(X_train, X_test, y_train, y_test):
    # print 'LDA'
    # LDA
    lda = LDA()

    t1 = datetime.now()
    lda.fit(X_train, y_train)
    t2 = datetime.now()
    y_pred = lda.predict(X_test)

    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)

    acuracia = accuracy_score(y_test, y_pred)
    # print('Accuracy' + str(accuracy_score(y_test, y_pred)))

    # # cria a matriz de confusao
    # cm = confusion_matrix(y_test, y_pred)
    # print cm
    #
    skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=y_pred, normalize=True,
                                        title="Matrix de Confusao")
    plt.tight_layout()
    plt.savefig("resultados/confusion_matrix_lda.pdf")
    # plt.show()

    # # scatter plot
    # plt.subplot(211)
    # plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train, s=25, edgecolor='k')
    # # plt.scatter(lda.support_vectors_[:, 0], lda.support_vectors_[:, 1], marker='x')
    # plt.savefig("scatter_lda.pdf")

    return [acuracia, t2-t1]

def logistic_regression(X_train, X_test, y_train, y_test):
    # print 'Logistic Regression'
    #fonte: https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
    # all parameters not specified are set to their defaults
    logisticRegr = LogisticRegression()

    t1 = datetime.now()
    logisticRegr.fit(X_train, y_train)
    t2 = datetime.now()
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
    skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=y_pred, normalize=True,
                                        title="Matrix de Confusao")
    plt.tight_layout()
    plt.savefig("resultados/confusion_matrix_lr.pdf")
    # plt.show()

    # scatter plot
    # plt.subplot(211)
    # plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train, s=25, edgecolor='k')
    # # plt.scatter(logisticRegr.support_vectors_[:, 0], logisticRegr.support_vectors_[:, 1], marker='x')
    # plt.savefig("scatter_lr.pdf")

    return [acuracia, t2-t1]


def GridSearch(X_train, y_train, k):
    # define range dos parametros
    C_range = 2. ** numpy.arange(-5, 15, 2)
    gamma_range = 2. ** numpy.arange(3, -15, -2)
    # k = ['rbf']
    # k = ['linear', 'rbf']
    param_grid = dict(gamma=gamma_range, C=C_range, kernel=k)

    # instancia o classificador, gerando probabilidades
    srv = svm.SVC(probability=True)

    # faz a busca
    grid = GridSearchCV(srv, param_grid, n_jobs=-1, verbose=True)
    grid.fit(X_train, y_train)

    # recupera o melhor modelo
    model = grid.best_estimator_

    # imprime os parametros desse modelo
    print grid.best_params_
    return model


def main(datatr, datats):
    ini = datetime.now()
    print ini

    acuracias = []
    tempo_total = []
    tempo_treino = []
    tempo_busca = []
    vetores = []
    matriz_confusao = []
    probabilidades = []


    # loads data
    print "Loading data..."
    X_train, y_train = load_svmlight_file(datatr)
    X_test, y_test = load_svmlight_file(datats)

    # X_data, y_data = load_svmlight_file(datatr)
    # X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.5, random_state=5)

    X_train = X_train.toarray()
    X_test = X_test.toarray()

    #X_train_dense = X_train.toarray()
    #X_test_dense = X_test.toarray()

    # X_train = X_train[0:1000]
    # y_train = y_train[0:1000]
    # splits data
    #print "Spliting data..."

    # cria um SVM
    tt1 = datetime.now()
    clf = svm.SVC(kernel='linear', C=1)

    # treina o classificador na base de treinamento
    # print "Training Classifier..."
    t1_svm_linear = datetime.now()
    clf.fit(X_train, y_train)
    t2_svm_linear = datetime.now()
    tempo_treino.append(t2_svm_linear - t1_svm_linear)

    # # mostra o resultado do classificador na base de teste
    acuracias.append(clf.score(X_test, y_test))
    #print clf.score(X_test, y_test)

    # predicao do classificador
    y_pred = clf.predict(X_test)

    # cria a matriz de confusao
    # cm = confusion_matrix(y_test, y_pred)
    # print cm

    vetores.append(clf.n_support_[0] + clf.n_support_[1])

    skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=y_pred, normalize=True,
                                        title="Matrix de Confusao")
    plt.tight_layout()
    plt.savefig("resultados/confusion_matrix_svm_linear.pdf")
    tt2 = datetime.now()
    tempo_total.append(tt2 - tt1)

    # GridSearch retorna o melhor modelo encontrado na busca
    tt1 = datetime.now()
    t1 = datetime.now()
    best = GridSearch(X_train, y_train, ['rbf'])
    t2 =datetime.now()
    tempo_busca.append(t2-t1)

    # Treina usando o melhor modelo
    t1 = datetime.now()
    best.fit(X_train, y_train)
    t2 = datetime.now()
    tempo_treino.append(t2-t1)
    # resultado do treinamento
    acuracia = best.score(X_test, y_test)
    acuracias.append(acuracia)

    # predicao do classificador
    y_pred = best.predict(X_test)

    # cria a matriz de confusao
    cm = confusion_matrix(y_test, y_pred)
    matriz_confusao.append(cm)
    # print cm

    #vetores de suporte
    vetores.append(best.n_support_[0] + best.n_support_[1])

    # probabilidades
    # probs = best.predict_proba(X_test)
    # print probs
    # probabilidades.append(probs)

    skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=y_pred, normalize=True,
                                        title="Matrix de Confusao")
    plt.tight_layout()
    plt.savefig("resultados/confusion_matrix_svm_rbf.pdf")

    tt2 = datetime.now()
    tempo_total.append(tt2-tt1)

    # ## classificadores anteriores

    #kNN
    a = datetime.now()
    res_knn = knn(X_train, X_test, y_train, y_test, 9, 'euclidean', 'distance')
    b = datetime.now()
    acuracias.append(res_knn[0])
    tempo_treino.append(res_knn[1])
    tempo_total.append(b - a)

    #Naive bayes
    a = datetime.now()
    res_nb = naive_bayes(X_train, X_test, y_train, y_test)
    b = datetime.now()
    acuracias.append(res_nb[0])
    tempo_treino.append(res_nb[1])
    tempo_total.append(b - a)

    #LDA
    a = datetime.now()
    res_lda = lda(X_train, X_test, y_train, y_test)
    b = datetime.now()
    acuracias.append(res_lda[0])
    tempo_treino.append(res_lda[1])
    tempo_total.append(b - a)

    # logistic regression
    a = datetime.now()
    res_lr = logistic_regression(X_train, X_test, y_train, y_test)
    b = datetime.now()
    acuracias.append(res_lr[0])
    tempo_treino.append(res_lr[1])
    tempo_total.append(b - a)

    salvar_arquivo('resultados/svm_results.csv', acuracias, tempo_busca, tempo_treino, tempo_total, vetores)
    fim = datetime.now()
    print 'tempo decorrido: ' + str(fim-ini)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Use: svm.py <tr> <ts>")
    main(sys.argv[1], sys.argv[2])
