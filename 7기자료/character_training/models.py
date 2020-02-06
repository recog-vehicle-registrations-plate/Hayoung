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
# import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K
# 랜덤시드 고정시키기
np.random.seed(3)
# 이미지데이터
img_dir = "/home/pirl/Downloads/license_plate/data/result"
categories = ["0", '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
              'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
              'T', 'U', 'V', 'W', 'X', 'Y']

# categories = ["0",'1','2','3','4','5','6','7','8','9']
nb_classes = len(categories)
print(nb_classes)
image_w = 12
image_h = 20
pixel = image_w * image_h * 3
X = []
y = []
for idx, char in enumerate(categories):
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    image_dir = img_dir + '/' + char
    files = glob.glob(image_dir + "/*.jpeg")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert('RGB')
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        X.append(data)
        y.append(label)
        if i % 7000 == 0:
            print(char, " : ", f)
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X,y)
xy = (X_train, X_test, y_train, y_test)
np.save("./lpchar.npy",xy)
print("ok", len(y))
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# history graph
'''
y_vloss = history.history['val_loss']
y_loss = history.history['loss']
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label='val_set_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='train_set_oss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()
'''