from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import csv
import sys

import numpy as np
import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import scikitplot as skplt

def criar_modelo_uma_camada(units=50):
    model = Sequential()
    model.add(Dense(units, activation='relu', input_dim=200))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def criar_modelo_duas_camadas(units=50):
    model = Sequential()
    model.add(Dense(units, activation='relu', input_dim=200))
    model.add(Dense(units, activation='relu', input_dim=200))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def gerar_resultados(iteracao, model, history, X_test, y_test, label):
    print(iteracao)
    resultado =[]
    resultado.append(iteracao)
    print (model.metrics_names)
    score = model.evaluate(X_test, y_test, batch_size=128)
    resultado.append(score)
    # print(score)

    y_pred = model.predict_classes(X_test)

    # matriz de confusao
    cm = confusion_matrix(label, y_pred)
    resultado.append(cm)
    # print(cm)
    skplt.metrics.plot_confusion_matrix(y_true=label, y_pred=y_pred, normalize=True,
                                        title="Matrix de Confusao ")
    plt.tight_layout()
    plt.savefig("resultados/cm_" + str(iteracao[0]) + '_' + str(iteracao[1]) + '_' + str(iteracao[2]) + '_' + str(iteracao[3]) + ".pdf")
    plt.close()
    # list all data in history
    # print(history.history.keys())
    # summarize history for accuracy

    resultado.append(history.history['acc'])
    resultado.append(history.history['val_acc'])
    return resultado

def plot(resultados):
    for res in resultados:
        plt.plot(res[3])
        plt.plot(res[4])
        plt.title('model accuracy ' + str(res[0]))
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        plt.savefig('resultados/' + str(res[0][0]) + '_' + str(res[0][1]) + '_' + str(res[0][2])+'_' + str(res[0][3]) + '.pdf')
        plt.close()

    #loss
    labels = []
    loss = []
    for res in resultados:
        labels.append(str(res[0][0]) + '_' + str(res[0][1]) + '_' + str(res[0][2]) + '_' + str(res[0][3]))
        loss.append(res[1][0])
    plt.plot(loss)
    ind = np.arange(len(labels))
    plt.xticks(ind, labels)
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('parameters')
    plt.savefig('resultados/' + str(res[0][0]) + '_loss.pdf')
    plt.close()

    #acuracias
    # print('acuracias')
    labels = []
    acuracias = []
    for res in resultados:
        print(str(res[0]), res[1][0])
        labels.append(str(res[0][0]) + '_' + str(res[0][1]) + '_' + str(res[0][2])+'_' + str(res[0][3]))
        acuracias.append(res[1][1])
    plt.plot(acuracias)
    ind = np.arange(len(labels))
    plt.xticks(ind, labels)
    plt.title('Acuracias')
    plt.ylabel('accuracy')
    plt.xlabel('parameters')
    plt.savefig('resultados/' + str(res[0][0]) + '_acuracias.pdf')
    plt.close()




def salvar_arquivo(arquivo, dados):
    with open('resultados/' + arquivo, mode='w') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        print(dados)
        for linha in dados:
            employee_writer.writerow(linha)


def main(tr, ts):
    neuronios = [1, 10, 25, 50, 75, 100]
    epocas = [1, 10, 50, 100, 150, 200]

    # inicio
    X_train, y_train = load_svmlight_file(tr)
    X_test, y_test = load_svmlight_file(ts)

    ## save for the confusion matrix
    label = y_test
    ## converts the labels to a categorical one-hot-vector
    y_train = keras.utils.to_categorical(y_train, num_classes=2)
    y_test = keras.utils.to_categorical(y_test, num_classes=2)

    # 1. A quantidade de neurônios na camada escondida
    res_neuronios = []
    for n in neuronios:
        model = criar_modelo_uma_camada(n)
        history = model.fit(X_train, y_train, validation_split=0.33, epochs=200, batch_size=128)
        res = gerar_resultados(['neuronios', n, 1, 200], model, history, X_test, y_test, label)
        res_neuronios.append(res)

    salvar_arquivo('neuronios.csv', res_neuronios)
    plot(res_neuronios)

    # 2. Quantidade de camadas escondidas
    res_camadas = []
    model = criar_modelo_uma_camada(50)
    history = model.fit(X_train, y_train, validation_split=0.33, epochs=200, batch_size=128)
    res = gerar_resultados(['camadas', 50, 1, 200], model, history, X_test, y_test, label)
    res_camadas.append(res)

    model = criar_modelo_duas_camadas(50)
    history = model.fit(X_train, y_train, validation_split=0.33, epochs=200, batch_size=128)
    res = gerar_resultados(['camadas', 50, 2, 200], model, history, X_test, y_test, label)
    res_camadas.append(res)


    salvar_arquivo('camadas.csv', res_neuronios)
    plot(res_camadas)

    # 3. Número de épocas de treinamento
    res_epocas = []
    model = criar_modelo_uma_camada(50)
    for e in epocas:
        history = model.fit(X_train, y_train, validation_split=0.33, epochs=e, batch_size=128)
        res = gerar_resultados(['epocas', 50, 1, e], model, history, X_test, y_test, label)
        res_epocas.append(res)

    salvar_arquivo('epocas.csv', res_epocas)
    plot(res_epocas)

    # 4. Overfitting da rede. Qual arquitetura (simples ou complexa) entra em overfitting com mais facilidade?
    simples = [[1, 1], [5, 5], [10, 10]]
    complexa = [[90, 190], [95, 195], [100, 200]]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Use: mlp2.py <tr> <ts>")
    main(sys.argv[1], sys.argv[2])
