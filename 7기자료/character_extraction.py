import cv2
import numpy as np
import math


def img_modify(input_image):
    height, width, channel = input_image.shape

    # 이미지 양쪽 새로 끝을 잘라줌, 컨투어 씌이는거 방지
    no_side_image = input_image[3:height-3, 5:width - 5]
    h, w, c = no_side_image.shape

    # Convert Image to Grayscale
    gray = cv2.cvtColor(no_side_image, cv2.COLOR_BGR2GRAY)

    # Maximize Contrast (Optional)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    # Adaptive Thresholding
    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    img_thresh = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )

    # Morphology
    # mph_chg = cv2.erode(img_thresh, None)
    mph_chg = cv2.dilate(img_thresh, None)
    # mph_chg = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, None)
    # mph_chg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, None)

    return mph_chg, h, w, c


def contour_cal(input_morph, height, width, channel):
    # Find Contours
    contours, _ = cv2.findContours(
        input_morph,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

    # Prepare Data
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

    # max(x)-min(x), max(y)-min(y) & 가로 세로 비율 구하기
    MIN_AREA = 10
    MAX_AREA = 1000
    # MIN_WIDTH, MIN_HEIGHT = 2, 8
    MIN_RATIO, MAX_RATIO = 0.3, 2.0

    # MIN_AREA = 80
    # MIN_WIDTH, MIN_HEIGHT = 2, 8
    # MIN_RATIO, MAX_RATIO = 0.25, 1.0

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
        if area > MIN_AREA and area < MAX_AREA and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
        #     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        temp_result = cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), \
                                    color=(255, 255, 255), thickness=1)

    # 사각형들의 좌표를 구해서 계산
    d_x = []
    d_y = []
    d_x_diagonal = []
    d_y_diagonal = []

    for d in possible_contours:
        x_diagonal = d['x'] + d['w']
        y_diagonal = d['y'] + d['h']

        d_x.append(d['x'])
        d_y.append(d['y'])

        d_x_diagonal.append(x_diagonal)
        d_y_diagonal.append(y_diagonal)

    # print('x좌표', d_x)
    # print('y좌표', d_y)
    # print('x대각좌표',d_x_diagonal)
    # print('y대각좌표',d_y_diagonal)

    d_x_total = d_x + d_x_diagonal
    d_y_total = d_y + d_y_diagonal

    # print(d_x_total)

    max_x = max(d_x_total)
    min_x = min(d_x_total)

    max_y = max(d_y_total)
    min_y = min(d_y_total)

    # print('min(x)=>',min(d_x_total),'min(y)=>',min(d_y_total))
    # print('max(x)=>',max(d_x_total),'max(y)=>',max(d_y_total))

    x_Difference = max(d_x_total) - min(d_x_total)
    y_Difference = max(d_y_total) - min(d_y_total)

    # print('max(x)-min(x)=>',x_Difference)
    # print('max(y)-min(y)=>',y_Difference)

    x_divided_y = x_Difference / y_Difference

    # print('가로 나누기 세로 =', x_divided_y)
    #     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
    #     cv2.rectangle(temp_result, (d['x'], d['y']), (d['x']+d['w'], d['y']+d['h']),
    #                   color=(255, 255, 255), thickness=1)
    #     cv2.rectangle(temp_result, (60 , 2), (73 , 19),
    #                   color=(255, 255, 255), thickness=1)

    return x_divided_y, min_y, width, y_Difference, max_y


def extract_chr(input_cal, min_y, width, y_difference, max_y, input_img):
    if input_cal > 3:
        x = 1
        y = min_y
        w = (width - 1)
        h = y_difference

        img_trim = input_img[y:y + h, x:x + w]
        print('한줄짜리다')

    else:
        x = 1
        y1 = min_y
        w = (width - 1)
        h1 = math.floor(y_difference / 2)

        y2 = math.ceil(y_difference / 2) + 2
        h2 = max_y - math.ceil(y_difference / 2)

        img_trim1 = input_img[y1:y1 + h1, x:x + w]
        img_trim2 = input_img[y2:y2 + h2, x:x + w]

        print('두줄짜리인가봄')

































#
# def convert_2_grays(input_image):
#     gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
#
#     # plt.figure(figsize=(12, 10))
#     # plt.imshow(gray, cmap='gray')
#
#     return gray
#
#
# # Optional definition
# def maximize_contrast(input_image):
#     structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#
#     imgTopHat = cv2.morphologyEx(input_image, cv2.MORPH_TOPHAT, structuringElement)
#     imgBlackHat = cv2.morphologyEx(input_image, cv2.MORPH_BLACKHAT, structuringElement)
#
#     imgGrayscalePlusTopHat = cv2.add(input_image, imgTopHat)
#     optional_gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
#
#     # plt.figure(figsize=(12, 10))
#     # plt.imshow(gray, cmap='gray')
#
#     return optional_gray
#
#
# def to_adaptive_thr(input_image):
#     img_blurred = cv2.GaussianBlur(input_image, ksize=(5, 5), sigmaX=0)
#
#     img_thresh = cv2.adaptiveThreshold(
#         img_blurred,
#         maxValue=255.0,
#         adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         thresholdType=cv2.THRESH_BINARY_INV,
#         blockSize=19,
#         C=9
#     )
#
#     # plt.figure(figsize=(12, 10))
#     # plt.imshow(img_thresh, cmap='gray')
#
#     return img_thresh
#
#
# def mor_close(input_thr):
#     dst4_morph_close = cv2.morphologyEx(input_thr, cv2.MORPH_CLOSE, None)
#
#     return dst4_morph_close
#
#
# def seek_contours(input_image):
#     contours, _ = cv2.findContours(
#         input_image,
#         mode=cv2.RETR_LIST,
#         method=cv2.CHAIN_APPROX_SIMPLE
#     )
#
#     # temp_result = np.zeros((40, 100, 3), dtype=np.uint8)
#     #
#     # cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))
#     #
#     # plt.figure(figsize=(12, 10))
#     # plt.imshow(temp_result)
#
#     return contours
#
#
# def prepare_data(contours):
#     temp_result = np.zeros((40, 100, 3), dtype=np.uint8)
#
#     contours_dict = []
#
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=1)
#
#         # insert to dict
#         contours_dict.append({
#             'contour': contour,
#             'x': x,
#             'y': y,
#             'w': w,
#             'h': h,
#             'cx': x + (w / 2),
#             'cy': y + (h / 2)
#         })
#
#     # plt.figure(figsize=(12, 10))
#     # plt.imshow(temp_result, cmap='gray')
#
#     return contours_dict
#
#
# def calculate_w_h(contours_dict):
#     MIN_AREA = 50
#     MAX_AREA = 500
#     # MIN_WIDTH, MIN_HEIGHT = 2, 8
#     MIN_RATIO, MAX_RATIO = 0.25, 1.0
#
#     # MIN_AREA = 80
#     # MIN_WIDTH, MIN_HEIGHT = 2, 8
#     # MIN_RATIO, MAX_RATIO = 0.25, 1.0
#
#     possible_contours = []
#
#     cnt = 0
#     for d in contours_dict:
#         area = d['w'] * d['h']
#         ratio = d['w'] / d['h']
#
#         #     if area > MIN_AREA and area < MAX_AREA \
#         #     and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
#         #     and MIN_RATIO < ratio < MAX_RATIO:
#         #         d['idx'] = cnt
#         #         cnt += 1
#         #         possible_contours.append(d)
#
#         # 가로,세로, 넓이 비율 삭제
#         if area > MIN_AREA and area < MAX_AREA and MIN_RATIO < ratio < MAX_RATIO:
#             d['idx'] = cnt
#             cnt += 1
#             possible_contours.append(d)
#
#     # visualize possible contours
#     # temp_result = np.zeros((height, width, channel), dtype=np.uint8)
#
#     d_x = []
#     d_y = []
#     d_x_diagonal = []
#     d_y_diagonal = []
#
#     for d in possible_contours:
#         x_diagonal = d['x'] + d['w']
#         y_diagonal = d['y'] + d['h']
#
#         d_x.append(d['x'])
#         d_y.append(d['y'])
#
#         d_x_diagonal.append(x_diagonal)
#         d_y_diagonal.append(y_diagonal)
#
#     # print('x좌표', d_x)
#     # print('y좌표', d_y)
#     # print('x대각좌표',d_x_diagonal)
#     # print('y대각좌표',d_y_diagonal)
#
#     d_x_total = d_x + d_x_diagonal
#     d_y_total = d_y + d_y_diagonal
#
#     # max_x = max(d_x_total)
#     # min_x = min(d_x_total)
#
#     max_y = max(d_y_total)
#     min_y = min(d_y_total)
#
#     # print('min(x)=>',min(d_x_total),'min(y)=>',min(d_y_total))
#     # print('max(x)=>',max(d_x_total),'max(y)=>',max(d_y_total))
#
#     x_Difference = max(d_x_total) - min(d_x_total)
#     y_Difference = max(d_y_total) - min(d_y_total)
#
#     # print('max(x)-min(x)=>',x_Difference)
#     # print('max(y)-min(y)=>',y_Difference)
#
#     x_divided_y = x_Difference / y_Difference
#
#     print('가로 나누기 세로 =', x_divided_y)
#     #     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
#     #     cv2.rectangle(temp_result, (d['x'], d['y']), (d['x']+d['w'], d['y']+d['h']),
#     #                   color=(255, 255, 255), thickness=1)
#     #     cv2.rectangle(temp_result, (60 , 2), (73 , 19),
#     #                   color=(255, 255, 255), thickness=1)
#
#     return x_divided_y, min_y, max_y, y_Difference
#
#
# def ext_char(input_image, x_divided_y, min_y, max_y, y_difference):
#     if x_divided_y > 3:
#         x = 1
#         y = min_y
#         w = 99
#         h = y_difference
#
#         img_trim = input_image[y:y + h, x:x + w]
#
#         cv2.imshow('', img_trim)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#         print('한줄짜리다')
#
#     else:
#         x = 1
#         y1 = min_y
#         w = 99
#         h1 = math.floor(y_difference / 2)
#
#         y2 = math.ceil(y_difference / 2) + 2
#         h2 = max_y - math.ceil(y_difference / 2)
#
#         img_trim1 = input_image[y1:y1 + h1, x:x + w]
#         img_trim2 = input_image[y2:y2 + h2, x:x + w]
#
#         cv2.imshow('', img_trim1)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#         cv2.imshow('', img_trim2)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#         print('두줄짜리인가봄')
