from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import cv2 as cv
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.constraints import MaxNorm as maxnorm
from tensorflow.keras.metrics import  MeanIoU

train_dataset = 'cats_dogs_dataset/train'
train_names = os.listdir(path=train_dataset)
imgs = []  # картинки
roibox = []  # координаты RoI
train_labels = []  # классы (кошка/собака)

for trains in train_names:
    if trains.endswith('.jpg'):
        imgs.append(cv.imread(train_dataset + '/' + trains))  # Собираем массив фотографий
        roibox.append(open(train_dataset + '/' + trains.replace('jpg', 'txt')).read().split(
            ' '))  # собираем массив с координатами ROI
for roi in roibox:
    if roi[0] == '1':
        train_labels.append(0)  # выбираем номера классов животных 1-кошка 2-собака
    else:
        train_labels.append(1)
    roi.pop(0)  # удаляем номер класа из RoI чтобы остались лишь координаты
for i in range(len(imgs)):
    imgs[i] = imgs[i][int(roibox[i][1]):int(roibox[i][3]), int(roibox[i][0]):int(roibox[i][2])]  # Обрезаем ROI
    imgs[i] = cv.resize(imgs[i], (32, 32))  # меняем размер изображения
imgs = np.asarray(imgs) / 255
(trainX, testX, trainY, testY) = train_test_split(imgs, train_labels, test_size=0.25,
                                                  random_state=42)  # делим на тренировочную и тестовую выборку
for i in range(len(trainX)):
    trainX[i] = trainX[i].astype('float32')
for i in range(len(testX)):
    testX[i] = testX[i].astype('float32')

trainY = to_categorical(trainY)
testY = to_categorical(testY)
class_num = testY.shape[1]
model = Sequential()

# ВХОДНОЙ СЛОЙ
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))  # исключающий слой для предотвращения переобучения
model.add(BatchNormalization())  # Пакетная нормализация
# СВЕРТОЧНЫЕ СЛОИ
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.2))
#ПЛОТНО СВЯЗАННЫЙ СЛОЙ
model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
#ВЫХОДНОЙ СЛОИ
model.add(Dense(class_num, activation='softmax'))

model.compile(loss='mse', optimizer='adam', metrics=[MeanIoU(num_classes=2)])# компилируем модель

model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=50, epochs=15)# тренируем

# model.save('model')# сохраняем
