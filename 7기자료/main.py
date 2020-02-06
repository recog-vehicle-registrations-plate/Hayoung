import cv2
import os
import Image_modify as Im

current_path = os.getcwd()

# test_path = 원본 사진이 들어가있는 폴더(회전시켜야 할 이미지 넣으면 됨니다)
test_path = "./jupyter_test_folder"
# print('현재 폴더 경로', current_path)
# print('테스트 폴더 경로', test_path)

# 파일 불러오기, list 에 넣기
image_list = []
# r=root, d=directories, f = files
for r, d, f in os.walk(test_path):
    for file in f:
        if '.jpg' or '.jpeg' or '.gif' or '.png' or '.pgm' in file:
            image_list.append(os.path.join(r, file))


# 여기서부터 시작
# 먼저 이미지들을 resize 시켜줘야 함니다.
for name in image_list:
    img_color = cv2.imread(name)
    # h, w, c = pre.img_shape(img_color)

    # shape must be (40, 100, 3)
    good_size = (40, 100, 3)
    if img_color.shape != good_size:
        img_color = Im.resize_img(img_color)
    else:
        pass
    h, w, c = img_color.shape

    # print(name)

    Pre_image = Im.preprocessing(img_color)

    ##################################################
    # 한 번 넣어보는 모폴로지 '팽창'
    dst2 = cv2.dilate(Pre_image, None)
    ##################################################


    rotated_image_list = Im.find_contours_and_rotate_image(dst2, h, w, c, img_color)

    # print('image list 들 =>', rotated_image_list)

    # 이미지 저장 및 Show image
    for image_file in rotated_image_list:
        path = './test_images_results'

        # showing_file_name = name.split('/')[2] 에서 숫자 2는 본인 머신에 맞추어서 바꿔줌
        # name 변수는 파일의 경로까지 저장되어 있으므로 / 기준으로 split 해주고 실제 이미지 이름만 가져온다.
        showing_file_name = name.split('/')[2]
        result_img_file_name = 'rotated' + '_' + str(showing_file_name)

        print('file_name =>', showing_file_name)
        print('rotating...')
        print('result_img_file_name =>', result_img_file_name, '\n')

        # show image, 이미지 한번 뜨면 숫자 0 눌러서 그 다음 이미지 뜨게해야 에러 안걸림니다
        cv2.imwrite(os.path.join(path, result_img_file_name), image_file)
        # cv2.imshow(showing_file_name, image_file)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()










# filelist = ['test.tif','test2.tif']
# for imagefile in filelist:
#     im=Image.open(imagefile)
#     box=(50, 50, 200, 200)
#     im_crop=im.crop(box)
#     im_crop.show()








# """ For test images in a folder """
# image_list, _, _ = files.get_files(test_path)
#
# result_folder = './result/'
# if not os.path.isdir(result_folder):
#     os.mkdir(result_folder)
#
#
# #
# for i in range():
#     img_color = cv2.imread('17.jpg')
#     h, w, c = pre.img_shape(img_color)
#     img_prepro = pre.img_preprocessing(img_color)
#     temp_result = pre.find_contours(img_prepro, h, w, c)
#
#     cv2.imshow("result3", temp_result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# #
# #
# #
# #
# #
# #
#
# test_path = os.path.abspath("./자동차번호판")
#
#
# """ For test images in a folder """
# image_list, _, _ = files.get_files(test_path)
#
# result_folder = './result/'
# if not os.path.isdir(result_folder):
#     os.mkdir(result_folder)
# #
# #
# #
# # # load data
# #     for k, image_path in enumerate(image_list):
# #         print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
# #         print(k, image_path, image_list, 'asdasdad')
# #
# #     for i in range():
# #     # img_color = cv2.imread('17.jpg')
# #     #     h, w, c = img_shape(img_color)
# #     #     img_prepro = img_preprocessing(img_color)
# #     #     temp_result = find_contours(img_prepro, h, w, c)
# #     #
# #     #     cv2.imshow("result3", temp_result)
# #     #     cv2.waitKey(0)
#     #     cv2.destroyAllWindows()
    #