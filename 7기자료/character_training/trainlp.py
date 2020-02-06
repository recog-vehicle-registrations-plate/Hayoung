# 0. 사용할 패키지 불러오기
import numpy as np
import glob, os
from PIL import Image
from keras.models import Sequential, load_model
# from keras.layers.convolutional import Conv2D
# from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K
from keras.optimizers import SGD,Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import numpy
import pandas as pd


# 랜덤시드 고정시키기
np.random.seed(3)
# 이미지데이터 (문자들이 들어있는 폴더)
img_dir = "/home/pirl/data/Nogada_Char/characters"
categories = ["0", '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
              'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
              'T', 'U', 'V', 'W', 'X', 'Y']

# categories = ['3']

nb_classes = len(categories)
print(nb_classes)
image_w = 12
image_h = 20
pixel = image_w * image_h * 3
X = []
y = []


def plot_history(history, result_dir):
    plt.plot(history.history['accuracy'], marker='.')
    plt.plot(history.history['val_accuracy'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['accuracy', 'val_accuracy'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\taccuracy\tval_loss\tval_accuracy\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()



# 일반화

X_train, X_test, y_train, y_test = np.load('./lpchar.npy', allow_pickle=True)
X_train = X_train.astype(float) /255
X_test = X_test.astype(float)/255
print(X_train.shape)
print(X_train.shape[0])
# 모델
# with K.tf_ops.device('/device:GPU:0'):
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu', padding = 'same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (5, 3), activation='relu', padding = 'same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 1), activation='relu', padding = 'same'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))



# gpu set
#modelFromGpu = multi_gpu_model(model, gpus=2)
lr = 0.001
# optimizer Adam
optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model_dir = './model'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model_path = model_dir + '/LP_keras.model'
checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=6)
# 모델 학습
history = model.fit(X_train,y_train, batch_size = 32, epochs = 1,
                    validation_data = (X_test, y_test), callbacks = [checkpoint, early_stopping])
model.summary()

if not os.path.exists('results/'):
    os.mkdir('results/')
plot_history(history, 'results/')
save_history(history, 'results/')

model_json = model.to_json()
# 저장할 모델명 - 짓고싶은대로 지으시면 댐
with open("results/JW_model.json", "w") as json_file :
    json_file.write(model_json)
# 저장할 웨이트
model.save_weights('results/JW__weight.h5')



# 모델 평가
model.summary()
print("정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))