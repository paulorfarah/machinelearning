#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-
import datetime
import sys
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import scikitplot as skplt

from sklearn import preprocessing
# import pylab as pl

from digits import load_images

#def main(data):
#def main(x1, x2, y1, y2):
def main(x, y, random_state, k, metric, weight):
    #print datetime.datetime.now()
    #resultados = []
    # for x in range(int(x1), int(x2)):
         #for y in range(int(y1), int(y2)):
            #parametros
            # x = 20
            # y = 10
            # test_size = 0.5
            # random_state = 5
            # k = 3
            # metric = 'euclidean'
            #algorithm = 'auto'
    test_size = 0.5
    # fout = open("features.txt", "w", buffering=1)
    # images = load_images('digits/data', fout, int(x), int(y))
    # fout.close
    # fout.flush()
    # print "Loading data..."

    try:
        X_data, y_data = load_svmlight_file("data/features.txt")
        # splits data
        # print "Spliting data..."
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=float(test_size), random_state=int(random_state))

        X_train = X_train.toarray()
        X_test = X_test.toarray()

        # fazer a normalizacao dos dados #######
        #scaler = preprocessing.MinMaxScaler()
        #X_train = scaler.fit_transform(X_train_dense)
        #X_test = scaler.fit_transform(X_test_dense)

        # cria um kNN
        a = datetime.datetime.now()
        print a
        neigh = KNeighborsClassifier(n_neighbors=int(k), metric=metric, weights=weight)

        # print 'Fitting knn'
        neigh.fit(X_train, y_train)
        b = datetime.datetime.now()

        # predicao do classificador
        # print 'Predicting...'
        y_pred = neigh.predict(X_test)


        # mostra o resultado do classificador na base de teste
        acuracia = neigh.score(X_test, y_test)
        # print acuracia

        print b
        print 'Tempo decorrido: ' + str(b-a)

        #res = {'x': x, 'y': y, 'acuracia': acuracia}
        #resultados.append(res)
        # cria a matriz de confusao
        cm = confusion_matrix(y_test, y_pred)
        print cm
        skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=y_pred, normalize=True,
                                            title="Matrix de Confusao")
        plt.tight_layout()
        plt.savefig("confusion_matrix_scikit_" + str(k) + ".pdf")
        plt.show()

        return acuracia
    except:
        print '[=> ERRO]: Erro ao ler arquivo features.txt para x='+str(x) + ' (' + str(sys.exc_info()) + ').'
        # pass
        # print 0
        return -1

    # print resultados
    # print datetime.datetime.now()


if __name__ == "__main__":
        # if len(sys.argv) != 8:
        #         sys.exit("Use: knn.py <x, y, test_size, random_state, k, metric, weight>")

        # main(sys.argv[1])
        #main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
        # import argparse, sys
        #
        # parser = argparse.ArgumentParser()
        #
        # #parser.add_argument('--x', help='X')
        # #parser.add_argument('--y', help='Y')
        # parser.add_argument('--random_state', help='random_state')
        # parser.add_argument('--k', help='k')
        # parser.add_argument('--metric', help='metric')
        # parser.add_argument('--weight', help='weight')
        # parser.add_argument('--instance', help='instance')
        #
        # args = parser.parse_args()
        #
        # index = args.instance.rfind("/")
        # instance = args.instance[index + 1:]
        # instanceX = instance.split("_")[0]
        # instanceY = instance.split("_")[1]
        #
        # main(instanceX, instanceY, args.random_state, args.k, args.metric, args.weight)
        # resultados = []
        # print datetime.datetime.now()
        # for x in range(1, 100, 5):
        #     for y in range(1, 100, 5):
        #         acuracia= main(x, y, 5, 3, 'euclidean', 'uniform')
        #         print str(x) + '; ' + str(y) + '; ' + str(acuracia)
        # print datetime.datetime.now()
        # print resultados

        # for k in range(1, 15):
        #     for m in [ 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean', 'mahalanobis']:
        #         acuracia = main(1, 1, 5, k, m, 'uniform')
        #         print str(1) + '; ' + str(1) + '; ' + str(k) + '; ' + m + '; ' + str(acuracia)
        #
        #         acuracia = main(46, 41, 5, k, m, 'uniform')
        #         print str(46) + '; ' + str(41) + '; ' + str(k) + '; ' + m + '; ' + str(acuracia)
        #

        ks = [1, 3, 5, 7, 9]
        for k in ks:
            print 'k=' + str(k)
            acuracia = main(3, 3, 5, k, 'euclidean', 'uniform')
            print acuracia

# pl.matshow(cm)
# pl.colorbar()
# pl.show()