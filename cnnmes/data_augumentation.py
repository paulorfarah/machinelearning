# -*- coding: utf-8 -*-

import os
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
entrada = 'data/'

for i in range(1, 1601):
	if os.path.exists(entrada + 'ad' + str(i).zfill(4) + '.tif'):
		imagem = load_img(entrada + 'ad' + str(i).zfill(4) + '.tif')
		imagem = img_to_array(imagem)
		imagem = np.expand_dims(imagem, axis=0)
		imgAug = ImageDataGenerator(rotation_range=40, width_shift_range=0.2,
									height_shift_range=0.2, zoom_range=0.2,
									fill_mode='nearest', horizontal_flip=True)
		imgGen = imgAug.flow(imagem, save_to_dir='aug', save_format='tif', save_prefix='gen_ad')
		counter = 0
		for (i, newImage) in enumerate(imgGen):
			counter += 1
			if counter == 10:
				break


for j in range(1, 753):
	if os.path.exists(entrada + 'td' + str(j).zfill(4)+'.tif'):
		imagem = load_img(entrada + 'td' + str(j).zfill(4)+'.tif')
		print("Gerando imagens a partir de: %s" % ('td' + str(j).zfill(4)))
		imagem = img_to_array(imagem)
		imagem = np.expand_dims(imagem, axis=0)
		imgAug = ImageDataGenerator(rotation_range=40, width_shift_range=0.2,
									height_shift_range=0.2, zoom_range=0.2,
									fill_mode='nearest', horizontal_flip=True)
		imgGen = imgAug.flow(imagem, save_to_dir='aug', save_format='tif', save_prefix='gen_td')
		counter = 0
		for (i, newImage) in enumerate(imgGen):
			counter += 1
			if counter == 10:
				break