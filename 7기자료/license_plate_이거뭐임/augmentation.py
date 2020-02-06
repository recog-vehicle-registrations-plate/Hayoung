from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import os
from os import chdir


datagen = ImageDataGenerator(
        rotation_range=20,
    # 지정된 각도 범위내에서 임의로 원본이미지를 회전시킵니다. 단위는 도이며, 정수형입니다. 예를 들어 90이라면 0도에서 90도 사이에 임의의 각도로 회전시킵니다.
        width_shift_range=0.2,
    # 지정된 수평방향 이동 범위내에서 임의로 원본이미지를 이동시킵니다. 수치는 전체 넓이의 비율(실수)로 나타냅니다. 예를 들어 0.1이고 전체 넓이가 100이면, 10픽셀 내외로 좌우 이동시킵니다.
        height_shift_range=0.3,
    # 지정된 수직방향 이동 범위내에서 임의로 원본이미지를 이동시킵니다. 수치는 전체 높이의 비율(실수)로 나타냅니다. 예를 들어 0.1이고 전체 높이가 100이면, 10픽셀 내외로 상하 이동시킵니다.
        shear_range=0.2,
    # 밀림 강도 범위내에서 임의로 원본이미지를 변형시킵니다. 수치는 시계반대방향으로 밀림 강도를 라디안으로 나타냅니다. 예를 들어 0.5이라면, 0.5 라이안내외로 시계반대방향으로 변형시킵니다.
        zoom_range=0.2,
    # 지정된 확대/축소 범위내에서 임의로 원본이미지를 확대/축소합니다. “1-수치”부터 “1+수치”사이 범위로 확대/축소를 합니다. 예를 들어 0.3이라면, 0.7배에서 1.3배 크기 변화를 시킵니다.
        horizontal_flip=True,
    # 수평방향으로 뒤집기를 합니다.
        vertical_flip=True,
    # 수직방향으로 뒤집기를 합니다.
        fill_mode='nearest')



out_path_dir = '/home/pirl/LP_detection_model/data_aug_result'
folder_list = os.listdir(out_path_dir)

# print(folder_list)
# print(len(folder_list))
# print(folder_list[0])
# print(folder_list[1])
# print(folder_list[2])
# print(folder_list[3],"========")
#
# path_char2 = '/home/pirl/Downloads/license_plate/data/raw' + '/' + folder_list[0]
# print(os.listdir(path_char2))


def makeAug(classifier, current_num):



    for i in range(1):
        path_char = '/home/pirl/LP_detection_model/Nogada_Char' + '/' + classifier

        # file_list = 폴더 안에 있는 파일들 이예염
        file_list = os.listdir(path_char)

        # 파일을 하나씩 불러와서 shape 를 구하기로 해염
        for j in range(len(file_list)):
            img_path = path_char + '/' + file_list[j]

            print("img_path===",img_path)
            img = load_img(img_path)
            x = img_to_array(img)  # (3, 150, 150) 크기의 NumPy 배열
            x = x.reshape((1,) + x.shape)  # (1, 3, 150, 150) 크기의 NumPy 배열

            k = 0
            # TODO save_to_dir 뒤에 있는 숫자를 위에서 폴더명 즉 1,2,3,4 이런걸 받아오고 대입 % 를 이용해서
            # TODO 마찬가지로 save_prefix 또한 바꿔줄것
            for batch in datagen.flow(x, batch_size=1, save_to_dir='/home/pirl/LP_detection_model/data_aug_result/%s'%classifier,
                                      save_prefix='%s'%classifier, save_format='jpeg'):
                k += 1
                if k > 10000-current_num:
                    break  # 이미지 num장을 생성하고 마칩니다



def main():

    # makeAug('bk',2186)
    makeAug('3', 282)
    # makeAug('4', 226)
    # makeAug('5', 221)
    # makeAug('6', 312)
    # makeAug('7', 166)
    # makeAug('8', 283)
    # makeAug('9', 304)
    #
    # makeAug('A', 53)
    # makeAug('B', 83)
    # makeAug('C', 71)
    # makeAug('D', 34)
    # makeAug('E', 78)
    # makeAug('F', 88)
    # makeAug('G', 13)
    #
    # makeAug('H', 54)
    # makeAug('J', 228)
    # makeAug('K', 72)
    # makeAug('L', 53)
    # makeAug('M', 31)
    # makeAug('N', 40)
    #
    # makeAug('O', 4)
    # makeAug('P', 92)
    # makeAug('Q', 22)
    # makeAug('R', 75)
    # makeAug('S', 65)
    # makeAug('T', 30)
    # makeAug('U', 30)
    # makeAug('V', 145)
    # makeAug('W', 72)
    # makeAug('X', 17)
    # makeAug('Y', 25)


main()