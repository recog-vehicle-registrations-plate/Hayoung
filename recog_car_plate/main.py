import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import os

#######GLOBAL VARIABLE####
img_ori=''; height=''; width=''; channel=''; gray=''; structuingElement=''; imgTopHat=''; imgBlackHat=''; imgGrayscalePlusTopHat=''
img_blurred=''; img_thresh='';contours=''
MAX_DIAG_MULTIPLYER=''; MAX_ANGLE_DIFF=''; MAX_AREA_DIFF=''; MAX_WIDTH_DIFF=''; MAX_HEIGHT_DIFF=''; MIN_N_MATCHED=''
contours_dict=[]
possible_contours=[]
matched_result_idx = []
matched_result = []
plate_imgs = []
plate_infos = []
plate_chars = []

plt.style.use('dark_background')

def read_input_image(fname):
    global img_ori, height, width, channel

    img_ori=cv2.imread(fname)
    height, width, channel = img_ori.shape

    #plt.figure(figsize=(2,3))
    #plt.imshow(img_ori, cmap='gray')




def convert_image_grayscale():
    global img_ori, gray

    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

    #plt.figure(figsize=(2,3))
    #plt.imshow(gray, cmap='gray')


def maximize_contrast_optional():
    global structuingElement, imgTopHat, imgBlackHat, imgGrayscalePlusTopHat, gray

    structuingElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuingElement)
    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuingElement)

    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    #plt.figure(figsize=(2, 3))
    #plt.imshow(gray, cmap='gray')



def adaptive_thresholding():
    global gray, img_blurred, img_thresh
    img_blurred = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=0)

    img_thresh = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=-5)

    #plt.figure(figsize=(2, 3))
    #plt.imshow(img_thresh, cmap='gray')


def find_contours():
    global img_thresh,height, width, channel,contours
    contours, _ = cv2.findContours(
        img_thresh,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

    #plt.figure(figsize=(2, 3))
    #plt.imshow(temp_result)


def prepare_data():
    global height, width, channel, contours, contours_dict

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2)

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

    #plt.figure(figsize=(2, 3))
    #plt.imshow(temp_result, cmap='gray')


def select_candidates_by_char_size():
    global possible_contours, contours_dict, height, width, channel, MIN_AREA, MIN_WIDTH,MIN_HEIGHT, MIN_RATIO, MAX_RATIO
    MIN_AREA = 0.4
    MIN_WIDTH, MIN_HEIGHT = 0.2, 0.2
    MIN_RATIO, MAX_RATIO = 0.25, 1.0

    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']

        if area > MIN_AREA \
                and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
                and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                      thickness=2)

    #plt.figure(figsize=(2, 3))
    #plt.imshow(temp_result, cmap='gray')


def find_chars(contour_list):
    global  MAX_DIAG_MULTIPLYER, MAX_ANGLE_DIFF, MAX_AREA_DIFF, MAX_WIDTH_DIFF, MAX_HEIGHT_DIFF, MIN_N_MATCHED,possible_contours,matched_result_idx

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
        #print("recursive point")
        recursive_contour_list = find_chars(unmatched_contour)
        #print(len(recursive_contour_list[0]))
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)
            ## 문제발생지점
            #print(idx)
            break
        #print("break point!!!")

        break

    return matched_result_idx


def select_candidates_by_arrangement_of_contours():

    global matched_result_idx, MAX_DIAG_MULTIPLYER, MAX_ANGLE_DIFF, MAX_AREA_DIFF, MAX_WIDTH_DIFF, MAX_HEIGHT_DIFF, MIN_N_MATCHED,possible_contours,height, width, channel, matched_result

    MAX_DIAG_MULTIPLYER = 5  # 5
    MAX_ANGLE_DIFF = 12  # 12.0
    MAX_AREA_DIFF = 0.5  # 0.5
    MAX_WIDTH_DIFF = 0.8
    MAX_HEIGHT_DIFF = 0.2
    MIN_N_MATCHED = 3  # 3


    #print("prev to call find_chars")
    result_idx = find_chars(possible_contours)

    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for r in matched_result:
        for d in r:
            #         cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']),
                          color=(255, 255, 255), thickness=2)

    #plt.figure(figsize=(12, 10))
    #plt.imshow(temp_result, cmap='gray')



def rotate_plate_images():
    global matched_result, plate_imgs, plate_infos, img_thresh, width, height, channel

    PLATE_WIDTH_PADDING = 1.3  # 1.3
    PLATE_HEIGHT_PADDING = 1.5  # 1.5
    MIN_PLATE_RATIO = 1
    MAX_PLATE_RATIO = 10


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

        img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))

        img_cropped = cv2.getRectSubPix(
            img_rotated,
            patchSize=(int(plate_width), int(plate_height)),
            center=(int(plate_cx), int(plate_cy))
        )

        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[
            0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            continue

        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })

        #plt.subplot(len(matched_result), 1, i + 1)
        #plt.imshow(img_cropped, cmap='gray')



def another_thresholding_to_fine_chars():
    global MIN_AREA, MIN_WIDTH, MIN_HEIGHT, MIN_RATIO, MAX_RATIO, plate_imgs, plate_chars

    longest_idx, longest_text = -1, 0

    for i, plate_img in enumerate(plate_imgs):
        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
        _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # find contours again (same as above)
        contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        plate_max_x, plate_max_y = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            area = w * h
            ratio = w / h

            if area > MIN_AREA \
                    and w > MIN_WIDTH and h > MIN_HEIGHT \
                    and MIN_RATIO < ratio < MAX_RATIO:
                if x < plate_min_x:
                    plate_min_x = x
                if y < plate_min_y:
                    plate_min_y = y
                if x + w > plate_max_x:
                    plate_max_x = x + w
                if y + h > plate_max_y:
                    plate_max_y = y + h

        img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]

        # img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
        # _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

        chars = pytesseract.image_to_string(img_result, lang='eng', config='--psm 7 --oem 0')

        result_chars = ''
        has_digit = False
        for c in chars:
            if ord('A') <= ord(c) <= ord('Z') or c.isdigit():
                if c.isdigit():
                    has_digit = True
                result_chars += c


        print(result_chars)
        plate_chars.append(result_chars)

        if has_digit and len(result_chars) > longest_text:
            longest_idx = i

        #plt.subplot(len(plate_imgs), 1, i + 1)
        #plt.imshow(img_result, cmap='gray')

        return result_chars

if __name__=="__main__":
    empty_cnt=0
    tot=0
    for dir in os.listdir("/images"):
        #print(dir)
        for fname in os.listdir("/home/pirl/PycharmProjects/recog_car_plate/images/"+dir):
            tot+=1 # checking total file number

            f='/home/pirl/PycharmProjects/recog_car_plate/images/'+dir+"/"+fname
            #print(f)

            read_input_image(f)
            #print("="*5,"read_input_image","="*5)
            convert_image_grayscale()
            #print("="*5,"convert_image_grayscale","="*5)
            maximize_contrast_optional()
            #print("="*5,"maximize_contrast_optional","="*5)
            adaptive_thresholding()
            #print("="*5,"adaptive_thresholding","="*5)
            find_contours()
            #print("="*5,"find_contours","="*5)
            prepare_data()
            #print("="*5,"prepare_data","="*5)
            select_candidates_by_char_size()
            #print("="*5,"select_candidates_by_char_size","="*5)
            select_candidates_by_arrangement_of_contours()
            #print("="*5,"select_candidates_by_arrangement_of_contours","="*5)
            rotate_plate_images()
            #print("="*5,"rotate_plate_images","="*5)
            result_chars=another_thresholding_to_fine_chars()
            #print("="*5,"another_thresholding_to_fine_chars","="*5)

            if not result_chars:  # string is empty
                #print("empty")
                empty_cnt+=1
            else:
                print(result_chars)


    print("Percentage of detection : ", (tot-empty_cnt)/tot)