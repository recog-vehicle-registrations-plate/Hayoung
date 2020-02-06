from PIL import Image
import os, glob, numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from keras.models import model_from_json
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



categories = ["0", '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
              'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
              'T', 'U', 'V', 'W', 'X', 'Y']
# 원본 caltech_dir = "/home/pirl/Downloads/lptest/test1"
caltech_dir = "/home/pirl/LP_detection_model/char_test"
image_w = 12
image_h = 20

pixels = image_h * image_w * 3

X = []
filenames = []
files = glob.glob(caltech_dir+"/*.*")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    filenames.append(f)
    X.append(data)


X = np.array(X)


json_file = open("/home/pirl/LP_detection_model/license_plate/results/model1205.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("/home/pirl/LP_detection_model/license_plate/results/w1205.h5")
print("Loaded model from disk")


lr = 0.001
# optimizer Adam
optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999)

loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


prediction = loaded_model.predict(X)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
cnt = 0


#이 비교는 그냥 파일들이 있으면 해당 파일과 비교. 카테고리와 함께 비교해서 진행하는 것은 _4 파일.
for i in prediction:
    pre_ans = i.argmax()  # 예측 레이블
    print(i,'이게뭐임2')
    print(pre_ans,'이게뭐임1')
    pre_ans_str = ''
    # if pre_ans == 0: pre_ans_str = "비행기"
    # elif pre_ans == 1: pre_ans_str = "불상"
    # elif pre_ans == 2: pre_ans_str = "나비"
    # else: pre_ans_str = "게"
    print(filenames[cnt], "의 결과", categories[pre_ans])
    # if i[0] >= 0.8 : print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
    # if i[1] >= 0.8: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"으로 추정됩니다.")
    # if i[2] >= 0.8: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
    # if i[3] >= 0.8: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
    cnt += 1
    # print(i.argmax()) #얘가 레이블 [1. 0. 0.] 이런식으로 되어 있는 것을 숫자로 바꿔주는 것.
    # 즉 얘랑, 나중에 카테고리 데이터 불러와서 카테고리랑 비교를 해서 같으면 맞는거고, 아니면 틀린거로 취급하면 된다.
    # 이걸 한 것은 _4.py에.