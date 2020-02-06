import cv2
import os
import character_extraction as ce
import math
import Image_modify as Im

current_path = os.getcwd()

# test_path = 원본 사진이 들어가있는 폴더(회전시켜야 할 이미지 넣으면 됨니다)
test_path = "./Image_Rotated/lpadd1_results/"
# print('현재 폴더 경로', current_path)
# print('테스트 폴더 경로', test_path)

# 파일 불러오기, list 에 넣기
image_list = []
# r=root, d=directories, f = files
for r, d, f in os.walk(test_path):
    for file in f:
        if '.jpg' or '.jpeg' or '.gif' or '.png' or '.pgm' in file:
            image_list.append(os.path.join(r, file))


# print(image_list)

for name in image_list:
    img_color = cv2.imread(name)

    img_morph, height, width, channel = ce.img_modify(img_color)
    calculated, min_y, w, y_difference, max_y = ce.contour_cal(img_morph, height, width, channel)

    print('이미지 이름', name)
    print(calculated)

    # 원본 그림을 계산한 y 축 좌표를 참고하여 자르자 & 두줄짜리는 없다고 가정
    _, trim_w, _ = img_color.shape


    # 한 줄 짜리라고 가정
    if calculated > 0:
        print('한줄짜리다')
        x = 1
        y = min_y
        w = (trim_w - 1)
        h = y_difference

        img_trim = img_color[y:y + h, x:x + w]

        # 높이를 20로 일정하게 맞춰줌
        _, width, _ = Im.img_shape(img_trim)
        height = 20
        dim = (width, height)
        resized_image = cv2.resize(img_trim, dim, interpolation=cv2.INTER_AREA)
        # 높이 12로 맞춰주기 완료

        path = './chr_images_results'
        showing_file_name = name.split('/')[-1]
        result_img_file_name = 'extracted' + '_' + str(showing_file_name)

        cv2.imwrite(os.path.join(path, result_img_file_name), resized_image)
        #
        # cv2.imshow(name, resized_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    else:
        print('두줄짜리인가봄')
        x = 1
        y1 = min_y
        w = (width - 1)
        h1 = math.floor(y_difference / 2)

        y2 = math.ceil(y_difference / 2) + 2
        h2 = max_y - math.ceil(y_difference / 2)

        img_trim1 = img_color[y1:y1 + h1, x:x + w]
        img_trim2 = img_color[y2:y2 + h2, x:x + w]

        cv2.imshow(name, img_trim1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow(name, img_trim2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()