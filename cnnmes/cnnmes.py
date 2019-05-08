import datetime

from keras_preprocessing.image import ImageDataGenerator
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import csv

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from PIL import Image
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import scikitplot as skplt

num_classes = 12

train_file = 'train.txt'
test_file = 'test.txt'

# input image dimensions
img_rows, img_cols = 64, 64


#==========================================================================

def load_images(image_paths, convert=False):

    x = []
    y = []
    for image_path in image_paths:

        path, label = image_path.split(' ')

        path= './aug/' + path

        if convert:
            image_pil = Image.open(path).convert('RGB')
        else:
            image_pil = Image.open(path).convert('L')

        img = np.array(image_pil, dtype=np.uint8)

        x.append(img)
        y.append([int(label)])


    x = np.array(x)
    y = np.array(y)

    if np.min(y) != 0:
        y = y-1

    return x, y


def load_dataset(train_file, test_file, resize, convert=False, size=(224,224)):
    arq = open(train_file, 'r')
    texto = arq.read()
    train_paths = texto.split('\n')

    print('Size : ', size)

    train_paths.remove('') #remove empty lines
    train_paths.sort()
    x_train, y_train = load_images(train_paths, convert)

    arq = open(test_file, 'r')
    texto = arq.read()
    test_paths = texto.split('\n')

    test_paths.remove('') #remove empty lines
    test_paths.sort()
    x_test, y_test = load_images(test_paths, convert)

    if resize:
        print ("Resizing images...")
        x_train = resize_data(x_train, size, convert)
        x_test = resize_data(x_test, size, convert)

    if not convert:
        x_train = x_train.reshape(x_train.shape[0], size[0], size[1], 1)
        x_test = x_test.reshape(x_test.shape[0], size[0], size[1], 1)


    print (np.shape(x_train))
    return (x_train, y_train), (x_test, y_test)

def resize_data(data, size, convert):
    if convert:
        data_upscaled = np.zeros((data.shape[0], size[0], size[1], 3))
    else:
        data_upscaled = np.zeros((data.shape[0], size[0], size[1]))
    for i, img in enumerate(data):
        large_img = cv2.resize(img, dsize=(size[1], size[0]), interpolation=cv2.INTER_CUBIC)
        data_upscaled[i] = large_img

    print (np.shape(data_upscaled))
    return data_upscaled

def criar_modelo(units=128):
    # create cnn model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())

    # model.add(Dense(units, activation='relu'))
    model.add(Dense(units, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # print cnn layers
    print('Network structure ----------------------------------')
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
        if hasattr(layer, 'output_shape'):
            print(layer.output_shape)
    print('----------------------------------------------------')

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # model = Sequential()
    # model.add(Dense(units, activation='relu', input_dim=200))
    # model.add(Dense(2, activation='softmax'))
    #
    # model.compile(loss='binary_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])
    return model

def gerar_resultados(iteracao, model, history, X_test, y_test, label):
    print(iteracao)
    print('-------------------------------------------------------------')
    resultado =[]
    resultado.append(iteracao)
    print (model.metrics_names)

    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test accuracy:', score[1])
    score = model.evaluate(X_test, y_test, batch_size=128)
    resultado.append(score)
    # print(score)

    # print model.predict_classes(x_test) #classes predicted
    # print model.predict_proba(x_test) #classes probability

    pred = []
    y_pred = model.predict_classes(x_test)
    for i in range(len(x_test)):
        pred.append(y_pred[i])

    # print(confusion_matrix(label, pred))
    # matriz de confusao
    cm = confusion_matrix(label, pred)
    resultado.append(cm)
    skplt.metrics.plot_confusion_matrix(y_true=label, y_pred=pred, normalize=True,
                                        title="Matrix de Confusao ")
    plt.tight_layout()
    plt.savefig("resultados/cm_" + str(iteracao[0]) + '_' + str(iteracao[1]) + '_' + str(iteracao[2]) + '_' + str(iteracao[3]) + ".pdf")
    plt.close()

    resultado.append(history.history['acc'])
    resultado.append(history.history['val_acc'])
    return resultado

def plot(resultados):
    # # print(history.history.keys())
    # # summarize history for accuracy
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

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

def data_augumentation():
    #https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    img = load_img('data/train/cats/cat.0.jpg')  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely
# ==========================================================================


ini = datetime.datetime.now()
print(ini)

print ("Loading database...")
# gray scale
# input_shape = (img_rows, img_cols, 1)
# (x_train, y_train), (x_test, y_test) = load_dataset(train_file, test_file, resize=True, convert=False, size=(img_rows, img_cols))

# rgb
input_shape = (img_rows, img_cols, 3)
(x_train, y_train), (x_test, y_test) = load_dataset(train_file, test_file, resize=True, convert=True, size=(img_rows, img_cols))

### save for the confusion matrix
label = []
for i in range(len(x_test)):
    label.append(y_test[i][0])

# normalize images
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print ('x_train shape:', x_train.shape)

print (x_train.shape[0], 'train samples')
print (x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# modelo
# variar neuronios
# neuronios = [10, 50, 100, 150, 200, 250]
# neuronios = [225, 250, 275, 300]
# neuronios = [275]
neuronios = [10]

res_neuronios = []
for n in neuronios:
    model = criar_modelo(n)
    # history = model.fit(X_train, y_train, validation_split=0.33, epochs=200, batch_size=128)
    # res = gerar_resultados(['neuronios', n, 1, 200], model, history, X_test, y_test, label)
    # res_neuronios.append(res)

    history = model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1, validation_data=(x_test, y_test))
    res = gerar_resultados(['n', n, 2, 175], model, history, x_test, y_test, label)
    res_neuronios.append(res)

    salvar_arquivo('neuronios.csv', res_neuronios)
    plot(res_neuronios)

neuronio = 120
acuracia = 0
for r in res_neuronios:
    if r[1][1] > acuracia:
        acuracia = r[1][1]
        neuronio = r[0][1]

print('neuronio: ' + str(neuronio))
print('acuracia: ' + str(acuracia))
# adicionar camada
# testar depois q achar melhor modelo

# variar épocas
# epocas = [10, 50, 100, 150, 200]
# epocas = [125, 150, 175]
# epocas = [175]
epocas = [20]
res_epocas = []

model = criar_modelo(neuronio)
for e in epocas:
    history = model.fit(x_train, y_train, validation_split=0.33, epochs=e, batch_size=128)
    res = gerar_resultados(['e', neuronio, 2, e], model, history, x_test, y_test, label)
    res_epocas.append(res)

salvar_arquivo('epocas.csv', res_epocas)
plot(res_epocas)


fim = datetime.datetime.now()
print(fim)
print(fim-ini)