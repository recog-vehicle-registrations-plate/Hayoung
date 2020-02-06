import cv2
import os
import numpy as np

current_path = os.getcwd()

# test_path = 원본 사진이 들어가있는 폴더(회전시켜야 할 이미지 넣으면 됨니다)
test_path = "./two_line_image_size"
# print('현재 폴더 경로', current_path)
# print('테스트 폴더 경로', test_path)

# 파일 불러오기, list 에 넣기
image_list = []
# r=root, d=directories, f = files
for r, d, f in os.walk(test_path):
    for file in f:
        if '.jpg' or '.jpeg' or '.gif' or '.png' or '.pgm' in file:
            image_list.append(os.path.join(r, file))

ht = []
wt = []

# 여기서부터 시작
# 먼저 이미지들을 resize 시켜줘야 함니다.
for name in image_list:
    img_color = cv2.imread(name)
    h, w, _ = img_color.shape

    ht.append(h)
    wt.append(w)

print(ht)
print(wt)

hht = np.array(ht)
hht_avg = np.mean(hht)

wwt = np.array(wt)
wwt_avg = np.mean(wwt)

print(hht_avg, wwt_avg)
