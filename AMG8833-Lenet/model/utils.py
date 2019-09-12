import cv2
import os
import time
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from random import shuffle

def amg8833_load_data():
	annotation_lines = []
	train = []
	test = []
	datagen = ImageDataGenerator(
		rotation_range=10,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest')

	path = './data/'
	directory = os.listdir(path)
	for directory_name in directory:
		#print(directory_name)
		file = os.listdir(path+directory_name+'/')
		#print(file)
		for filename in file:
			annotation_lines.append([path+directory_name+'/'+filename, directory_name])
	#print(annotation_lines)
	shuffle(annotation_lines)

	for i in range(len(annotation_lines)):
		img	= cv2.imread(annotation_lines[i][0], cv2.IMREAD_GRAYSCALE) 	# 灰度图读取
		img_type = img.shape

		#cv2.imshow('test.jpg',cv2.resize(256,256,img))
		#cv2.waitKey(0)
		img = img.reshape(1,img_type[0],img_type[1],1)	# 1: 图片的数量	2：图高	3：图宽	4：灰度维
		img = datagen.flow(img, batch_size=1)
		img = img.next().reshape(img_type[0],img_type[1])
		train.append(img)
		test.append(int(annotation_lines[i][1]))


	X_train = np.array(train[:int(len(train)*0.8)])		# 取全部数据的 80% 作为训练集
	Y_train = np.array(test[:int(len(test)*0.8)])	
	X_test = np.array(train[int(len(train)*0.8):])
	Y_test = np.array(test[int(len(test)*0.8):])

	return (X_train, Y_train), (X_test, Y_test)





if __name__ == '__main__':
	amg8833_load_data()