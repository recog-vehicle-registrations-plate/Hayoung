import cv2
import numpy as np

# borrowed from https://github.com/kairess/license_plate_recognition


def img_shape(input_img):
    height, width, channel = input_img.shape

    return height, width, channel


def resize_img(input_img):
    height = 40
    width = 100
    dim = (width, height)
    resized_image = cv2.resize(input_img, dim, interpolation=cv2.INTER_AREA)
    # print('Resized Dimensions : ', resized_image.shape)

    return resized_image


def preprocessing(input_img):
    img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    mor_di = cv2.dilate(img_gray, None)
    # mor_close = cv2.morphologyEx(mor_di, cv2.MORPH_CLOSE, None)
    _, thr1 = cv2.threshold(mor_di, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return thr1


def find_contours_and_rotate_image(input_img, height, width, channel, original_image):

    # Contours 찾기
    contours, hierarchy = cv2.findContours(input_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

    # Contours 를 기준으로 사각형 영역 찾기
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    contours_dict = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=1)

        # insert to dict
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })

    # 기준점을 세운후 사각형 영역 필터팅
    MIN_AREA = 100
    MAX_AREA = 2000
    # MIN_WIDTH, MIN_HEIGHT = 2, 8
    MIN_RATIO, MAX_RATIO = 1.0, 2.0

    possible_contours = []

    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']

        #     if area > MIN_AREA and area < MAX_AREA \
        #     and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
        #     and MIN_RATIO < ratio < MAX_RATIO:
        #         d['idx'] = cnt
        #         cnt += 1
        #         possible_contours.append(d)

        # 가로,세로, 넓이 비율 삭제
        if area > MIN_AREA and area < MAX_AREA:
            # and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
        #     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        temp_result = cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), \
                                    color=(255, 255, 255), thickness=1)



    # 사각형 영역 걸러내기
    MAX_DIAG_MULTIPLYER = 5  # 5
    MAX_ANGLE_DIFF = 50.0  # 12.0
    MAX_AREA_DIFF = 3  # 0.5
    MAX_WIDTH_DIFF = 2
    MAX_HEIGHT_DIFF = 0.5
    MIN_N_MATCHED = 2  # 3

    def find_chars(contour_list):
        matched_result_idx = []

        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']:
                    continue

                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])

                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

                distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                if dx == 0:
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))
                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                width_diff = abs(d1['w'] - d2['w']) / d1['w']
                height_diff = abs(d1['h'] - d2['h']) / d1['h']

                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                        and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                        and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                    matched_contours_idx.append(d2['idx'])

            # append this contour
            matched_contours_idx.append(d1['idx'])

            if len(matched_contours_idx) < MIN_N_MATCHED:
                continue

            matched_result_idx.append(matched_contours_idx)

            unmatched_contour_idx = []
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4['idx'])

            unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

            # recursive
            recursive_contour_list = find_chars(unmatched_contour)

            for idx in recursive_contour_list:
                matched_result_idx.append(idx)

            break

        return matched_result_idx

    result_idx = find_chars(possible_contours)

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for r in matched_result:
        for d in r:
            #         cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            temp_result = cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']),
                                        color=(255, 255, 255), thickness=1)


    # Rotate Image
    PLATE_WIDTH_PADDING = 1.3  # 1.3
    PLATE_HEIGHT_PADDING = 1.5  # 1.5
    MIN_PLATE_RATIO = 1
    MAX_PLATE_RATIO = 10

    plate_imgs = []
    plate_infos = []


    image_rotated_list = []



    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )

        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

        img_rotated = cv2.warpAffine(original_image, M=rotation_matrix, dsize=(width, height))
        image_rotated_list.append(img_rotated)

        # img_cropped = cv2.getRectSubPix(
        #     img_rotated,
        #     patchSize=(int(plate_width), int(plate_height)),
        #     center=(int(plate_cx), int(plate_cy))
        # )
        #
        # if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[
        #     0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
        #     continue
        #
        # plate_imgs.append(img_cropped)
        # plate_infos.append({
        #     'x': int(plate_cx - plate_width / 2),
        #     'y': int(plate_cy - plate_height / 2),
        #     'w': int(plate_width),
        #     'h': int(plate_height)
        # })

    return image_rotated_list



# if __name__ == '__main__':
#
#     # load data
#     for k, image_path in enumerate(image_list):
#         print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
#         print(k,image_path,image_list,'asdasdad')
#         image = imgproc.loadImage(image_path)
#
#         bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
#
#         # save score text
#         filename, file_ext = os.path.splitext(os.path.basename(image_path))
#         mask_file = result_folder + "/res_" + filename + '_mask.jpg'
#         cv2.imwrite(mask_file, score_text)
#
#         file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)
#
#     print("elapsed time : {}s".format(time.time() - t))
#
#
#
#
#
#
#     for i in range()
#     img_color = cv2.imread('17.jpg')
#     h, w, c = img_shape(img_color)
#     img_prepro = img_preprocessing(img_color)
#     temp_result = find_contours(img_prepro, h, w, c)
#
#     cv2.imshow("result3", temp_result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
#
# # def find_contours(input_img):
# #     contours, hierarchy = cv2.findContours(input_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #
# #     # for cnt in contours:
# #     # cv.drawContours(img_color, [cnt], 0, (255, 0, 0), 3)  # blue
# #     # cv.imshow("result1", img_color)
# #     # cv.waitKey(0)
# #     #
# #     # for cnt in contours:
# #     #    x, y, w, h = cv2.boundingRect(cnt)
# #     #    cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
# #     # cv2.imshow("result2", img_color)
# #     # cv2.waitKey(0)
# #
# #     for cnt in contours:
# #         rect = cv2.minAreaRect(cnt)
# #         box = cv2.boxPoints(rect)
# #         box = np.int0(box)
# #         cv2.drawContours(img_color,[box],0,(0,0,255),2)
# #     cv2.imshow("result3", img_color)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()