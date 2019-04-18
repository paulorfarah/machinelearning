#!/usr/bin/python

#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_svmlight_file, make_classification, make_gaussian_quantiles, make_blobs
# from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

import csv

def salvar_arquivo(arquivo, acuracia, vetores, estatisticas):
    # abrindo o arquivo para escrita
    with open(arquivo, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(acuracia)
        spamwriter.writerow(vetores)
        spamwriter.writerow(estatisticas)


def GridSearch(X_train, y_train):
    # define range dos parametros
    C_range = 2. ** numpy.arange(-5, 15, 2)
    gamma_range = 2. ** numpy.arange(3, -15, -2)
    k = ['rbf']
    # k = ['linear', 'rbf']
    param_grid = dict(gamma=gamma_range, C=C_range, kernel=k)

    # instancia o classificador, gerando probabilidades
    srv = svm.SVC(probability=True)

    # faz a busca
    grid = GridSearchCV (srv, param_grid, n_jobs=-1, verbose=True)
    grid.fit (X_train, y_train)

    # recupera o melhor modelo
    model = grid.best_estimator_

    # imprime os parametros desse modelo
    print (grid.best_params_)
    return model


def main():
    acuracia_linear = []
    vetores_linear = []
    acuracia_brf = []
    vetores_brf = []

    for i in range(30):
        ## create data...
        plt.figure(figsize=(8, 8))

        # X_train, y_train = make_blobs(n_samples=300, centers=2)
        X_train, y_train = make_gaussian_quantiles(n_samples =300, n_features=2, n_classes =2)

        # cria um SVM
        clf = svm.SVC(kernel='linear')

        # treina o classificador na base de treinamento
        # print "Training Classifier..."
        clf.fit(X_train, y_train)

        plt.subplot(211)
        plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train, s=25, edgecolor='k')
        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], marker='x')

        # mostra o resultado do classificador na base de teste
        # print ('Desempenho Kernel Linear')

        # print (clf.score(X_train, y_train))
        acuracia_linear.append(clf.score(X_train, y_train))
        # print ('Vetores de suporte:',  clf.n_support_[0]+clf.n_support_[1] )
        vetores_linear.append(clf.n_support_[0] + clf.n_support_[1])

        # GridSearch retorna o melhor modelo encontrado na busca
        best = GridSearch(X_train, y_train)

        # resultado do treinamento
        # print ('Accuracia no kernl RBF:', )
        # print (best.score(X_train, y_train))
        acuracia_brf.append(best.score(X_train, y_train))
        # print ('Vetores de suporte:',  best.n_support_[0]+best.n_support_[1] )
        vetores_brf.append(best.n_support_[0]+best.n_support_[1])

        # Treina usando o melhor modelo
        best.fit(X_train, y_train)
        plt.subplot(212)
        plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train, s=25, edgecolor='k')
        plt.scatter(best.support_vectors_[:, 0], best.support_vectors_[:, 1], marker='x')
        plt.savefig('resultados/svm_'+ str(i) + '.eps')
        # plt.show()

        # resultado do treinamento
        # print ('Accuracia no teste:', )
        # print best.score(X_test, y_test)

        # predicao do classificador
        # y_pred = best.predict(X_test)

        # cria a matriz de confusao
        # cm = confusion_matrix(y_test, y_pred)
        # print cm


    # linear - acuracia
    maior_ac_li = numpy.max(acuracia_linear)
    menor_ac_li = numpy.min(acuracia_linear)
    media_ac_li = numpy.mean(acuracia_linear)
    median_ac_lin = numpy.median(acuracia_linear)
    dp_ac_lin = numpy.std(acuracia_linear)

    # linear vetores
    maior_vet_li = numpy.max(vetores_linear)
    menor_vet_li = numpy.min(vetores_linear)
    media_vet_li = numpy.mean(vetores_linear)
    median_vet_lin = numpy.median(vetores_linear)
    dp_vet_lin = numpy.std(vetores_linear)
    estatistica_li = [maior_ac_li, menor_ac_li, media_ac_li, median_ac_lin, dp_ac_lin, maior_vet_li, menor_vet_li, media_vet_li, median_vet_lin, dp_vet_lin]
    salvar_arquivo('resultados/svm_linear.csv', acuracia_linear, vetores_linear, estatistica_li)

    # brf - acuracia
    maior_ac_rbf = numpy.max(acuracia_brf)
    menor_ac_rbf = numpy.min(acuracia_brf)
    media_ac_brf = numpy.mean(acuracia_brf)
    median_ac_brf = numpy.median(acuracia_brf)
    dp_ac_brf = numpy.std(acuracia_brf)

    # brf - vetores
    maior_vet_brf = numpy.max(vetores_brf)
    menor_vet_brf = numpy.min(vetores_brf)
    media_vet_brf = numpy.mean(vetores_brf)
    median_vet_brf = numpy.median(vetores_brf)
    dp_vet_brf = numpy.std(vetores_brf)
    estatistica_brf = [maior_ac_rbf, menor_ac_rbf, media_ac_brf, median_ac_brf, dp_ac_brf, maior_vet_brf, menor_vet_brf, media_vet_brf, median_vet_brf, dp_vet_brf]
    salvar_arquivo('resultados/svm_rbf.csv', acuracia_brf, vetores_brf, estatistica_brf)




if __name__ == "__main__":
    if len(sys.argv) != 1:
        sys.exit("Use: svmtoy.py")

    main()
