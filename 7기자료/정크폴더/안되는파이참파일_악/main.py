import cv2
import os
import Image_rotated as Im
import character_extraction as che

current_path = os.getcwd()

# test_path = 원본 사진이 들어가있는 폴더(회전시켜야 할 이미지 넣으면 됨니다)
test_path = "./test_images"
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
    h, w, c = Im.img_shape(img_color)

    # print(name)

    Pre_image = Im.preprocessing(img_color)
    rotated_image_list = Im.find_contours_and_rotate_image(Pre_image, h, w, c, img_color)

    # print('image list 들 =>', rotated_image_list)

    # 이미지 저장 및 Show image
    for image_file in rotated_image_list:
        path = './results'

        # showing_file_name = name.split('/')[2] 에서 숫자 2는 본인 머신에 맞추어서 바꿔줌
        # name 변수는 파일의 경로까지 저장되어 있으므로 / 기준으로 split 해주고 실제 이미지 이름만 가져온다.
        showing_file_name = name.split('/')[2]
        result_img_file_name = 'rotated' + '_' + str(showing_file_name)

        print('file_name =>', showing_file_name)
        print('rotating...')
        print('result_img_file_name =>', result_img_file_name, '\n')

        # show image, 이미지 한번 뜨면 숫자 0 눌러서 그 다음 이미지 뜨게해야 에러 안걸림니다
        cv2.imwrite(os.path.join(path, result_img_file_name), image_file)
        cv2.imshow(showing_file_name, image_file)
        cv2.waitKey(0)
        cv2.destroyAllWindows()











#
# current_path = os.getcwd()
#
# # test_path = 원본 사진이 들어가있는 폴더(회전시켜야 할 이미지 넣으면 됨니다)
# test_path = "./one_two_line"
# # print('현재 폴더 경로', current_path)
# # print('테스트 폴더 경로', test_path)
# # 파일 불러오기, list 에 넣기
# image_list = []
# # r=root, d=directories, f = files
# for r, d, f in os.walk(test_path):
#     for file in f:
#         if '.jpg' or '.jpeg' or '.gif' or '.png' or '.pgm' in file:
#             image_list.append(os.path.join(r, file))
#
# print('가져온 이미지 리트스=>', image_list)
#
# # 여기서부터 시작
# # 먼저 이미지들을 1줄, 2줄 짜리로 나눠줍니다. 2줄짜리는 정사각형의 형태를 띄므로 가로 세로 비율이 비슷한 것은 다 두줄입니다.
# for name in image_list:
#     img_color = cv2.imread(name)
#     h, w, c = Im.img_shape(img_color)
#
#     if h*2 < w:
#         print(name, '한줄짜리')
#         # shape must be (40, 100, 3)
#         img_color, h, w = Im.resize_one_line_img(img_color)
#
#     else:
#         print(name, '한줄짜리')
#         # 글자를 홀쭉하게 변경 시켜야 함
#         img_color, h, w = Im.resize_two_line_img(img_color)
#
#     temp_image = Im.chg_form(img_color)
#     temp_contour = Im.find_contours(temp_image)
#     temp_rec_contour = Im.find_rectangle(temp_contour)
#     temp_fil_rec = Im.filter_rectangle(temp_rec_contour)
#     temp_result = Im.find_lp_rectangle(temp_fil_rec)
#     rotated_image_list = Im.rotate_image(img_color, temp_result)
#
#
#
#     # rotated_image_list = Im.find_contours_and_rotate_image(temp_image, h, w, 3, img_color)
#
#     # print("rotated_image_listrotated_image_listrotated_image_listrotated_image_listrotated_image_list ===== ", rotated_image_list)
#
#     # 이미지 저장 및 Show image
#     for image_file in rotated_image_list:
#         path = './results'
#
#         # name 변수는 파일의 경로까지 저장되어 있으므로 / 기준으로 split 해주고 실제 이미지 이름만 가져온다.
#         showing_file_name = name.split('/')[-1]
#         result_img_file_name = 'rotated' + '_' + str(showing_file_name)
#
#         print('file_name =>', showing_file_name)
#         print('rotating...')
#         print('result_img_file_name =>', result_img_file_name, '\n')
#
#         # 이미지 저장하기 (필수아님, 어떻게 저장되었는지 보고 싶으면 출력)
#         print(image_list)
#         cv2.imwrite(os.path.join(path, result_img_file_name), image_file)
#
#         # 이미지 보여주기
#         # 이미지 한번 뜨면 숫자 0 눌러서 그 다음 이미지 뜨게해야 에러 안걸림니다
#         cv2.imshow(showing_file_name, image_file)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


        # 저장된 이미지의 가로, 세로 비율을 구해서 글자부분만 추출
        # rotated_image_list 에는 회전된 이미지들이 들어가 있다

        # for my_images in rotated_image_list:
        #     gray_img = che.convert_2_grays(my_images)
        #     max_image = che.maximize_contrast(gray_img)
        #     ad_thr = che.to_adaptive_thr(max_image)
        #     sk_contour = che.seek_contours(ad_thr)
        #
        #     pre_data = che.prepare_data(sk_contour)
        #     cla, miny, maxy, y_diff = che.calculate_w_h(pre_data)
        #
        #     # 캐릭터 부분만 뽑아내기
        #     image_char_ext = che.ext_char(my_images, cla, miny, maxy, y_diff)
        #
        #     print(len(my_images))













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