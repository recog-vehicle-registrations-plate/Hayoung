import cv2
import os
import character_extraction as ce
import math
import Image_modify as Im

current_path = os.getcwd()

# test_path = 슬라이딩 윈도우를 해서 읽어낼 글자가 있는 것을 가져오자
test_path = "./chr_images_results/"
# print('현재 폴더 경로', current_path)
# print('테스트 폴더 경로', test_path)

# 파일 불러오기, list 에 넣기
image_list = []
# r=root, d=directories, f = files
for r, d, f in os.walk(test_path):
    for file in f:
        if '.jpg' or '.jpeg' or '.gif' or '.png' or '.pgm' in file:
            image_list.append(os.path.join(r, file))

for name in image_list:
    img_color = cv2.imread(name)

    height, width, _ = Im.img_shape(img_color)
    # height = 12
    # dim = (width, height)
    # resized_image = cv2.resize(img_color, dim, interpolation=cv2.INTER_AREA)

    print(name)
    print(height, width)
    print('\n')

