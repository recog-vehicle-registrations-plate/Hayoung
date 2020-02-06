import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pytesseract

conf = ('--tessdata-dir "/usr/share/tesseract-ocr/4.00/tessdata" --oem 2 --psm 7')

def histogram_of_pixel_projection(img):
    """
    This method is responsible for licence plate segmentation with histogram of pixel projection approach
    :param img: input image
    :return: list of image, each one contain a digit
    """
    # list that will contains all digits
    caracrter_list_image = list()

    # img = crop(img)

    # Add black border to the image
    BLACK = [0, 0, 0]
    img = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=BLACK)

    # change to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """
    add Morphological Operation- Erosion
    
    """
    kernel = np.ones((5,5), np.uint8)
    plt.subplot(131)
    plt.title("before Erosion")
    plt.imshow(img)
    #plt.show()
    # gray = cv2.erode(gray, kernel, iterations=1)
    # temp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # plt.subplot(132)
    # plt.title('after Erosion')
    # plt.imshow(temp)


    # Change to numpy array format
    nb = np.array(gray)

    nb=cv2.erode(nb, kernel, iterations=1)

    # Binarization
    nb[nb >= 65] = 255
    nb[nb < 65] = 0


    plt.subplot(132)
    plt.xlabel(pytesseract.image_to_string(nb,lang='eng', config=conf))
    plt.title("Erosion and Binarization")
    plt.imshow(nb)

    # compute the sommation
    x_sum = cv2.reduce(nb, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    y_sum = cv2.reduce(nb, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

    # rotate the vector x_sum
    x_sum = x_sum.transpose()

    # get height and weight
    x = gray.shape[1]
    y = gray.shape[0]

    # division the result by height and weight
    x_sum = x_sum / y
    y_sum = y_sum / x

    # x_arr and y_arr are two vector weight and height to plot histogram projection properly
    x_arr = np.arange(x)
    y_arr = np.arange(y)

    # convert x_sum to numpy array
    z = np.array(x_sum)

    # convert y_arr to numpy array
    w = np.array(y_sum)

    # convert to zero small details
    z[z < 15] = 0
    z[z > 15] = 1

    # convert to zero small details and 1 for needed details
    w[w < 20] = 0
    w[w > 20] = 1


    # vertical segmentation
    test = z.transpose() * nb

    # horizontal segmentation
    test = w * test

    # plot histogram projection result using pyplot
    horizontal = plt.plot(w, y_arr)
    vertical = plt.plot(x_arr, z)
    #
    # plt.show(horizontal)
    # plt.show(vertical)

    f = 0
    ff = z[0]
    t1 = list()
    t2 = list()
    for i in range(z.size):
        if z[i] != ff:
            f += 1
            ff = z[i]
            t1.append(i)
    rect_h = np.array(t1)

    f = 0
    ff = w[0]
    for i in range(w.size):
        if w[i] != ff:
            f += 1
            ff = w[i]
            t2.append(i)
    rect_v = np.array(t2)
    # take the appropriate height

    # rectv = []
    # rectv.append(rect_v[0])
    # rectv.append(rect_v[1])
    # max = int(rect_v[1]) - int(rect_v[0])
    # for i in range(len(rect_v) - 1):
    #     diff2 = int(rect_v[i + 1]) - int(rect_v[i])
    #
    #     if diff2 > max:
    #         rectv[0] = rect_v[i]
    #         rectv[1] = rect_v[i + 1]
    #         max = diff2
    #
    # # extract character
    # for i in range(len(rect_h) - 1):
    #
    #     # eliminate slice that can't be a digit, a digit must have width bigger then 8
    #     diff1 = int(rect_h[i + 1]) - int(rect_h[i])
    #
    #     if (diff1 > 5) and (z[rect_h[i]] == 1):
    #         # cutting nb (image) and adding each slice to the list caracrter_list_image
    #         caracrter_list_image.append(nb[int(rectv[0]):int(rectv[1]), rect_h[i]:rect_h[i + 1]])
    #
    #         # draw rectangle on digits
    #         cv2.rectangle(img, (rect_h[i], rectv[0]), (rect_h[i + 1], rectv[1]), (0, 255, 0), 1)
    #         #cv2.rectangle(gray, (rect_h[i], rectv[0]), (rect_h[i + 1], rectv[1]), (0, 255, 0), 1)

    # Show segmentation result
    plt.subplot(133)
    image = plt.imshow(img)
    plt.title("Result image")
    #image = plt.imshow(gray)
    plt.show()

    return caracrter_list_image


if __name__ == '__main__':

    test_path = "./sample/ESRGAN_samples"
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

        print(img_color.shape)

        histo_chr = histogram_of_pixel_projection(img_color)
        print('???', histo_chr)
