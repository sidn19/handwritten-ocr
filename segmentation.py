import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from scipy.signal import argrelmin

import os

os.chdir(os.path.dirname(__file__))


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y

def hard_seggregate(arr, min_cutoff):
    N=len(arr)
    scan_array = False
    block_start = block_end = 0

    blocks = []

    for i in range(N):
        pixel_count_column = arr[i]
        if scan_array:
            if pixel_count_column < min_cutoff or i == N - 1:
                block_end = i
                scan_array = False
                blocks.append((block_start, block_end))
        else:
            if pixel_count_column >= min_cutoff:
                scan_array = True
                block_start = i
    
    return blocks

def smooth_seggregate(arr):
    pass


def get_lines_y_coordinates(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(
        gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Appplying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    freq = np.array([sum(row) // 255 for row in dilation])

    # freq = np.sum(dilation,axis=1,keepdims=True) / 255
    # print(list(freq))

    # smoothed = smooth(freq, 10)

    # mins = argrelmin(smoothed, order=2)
    # arr_mins = np.array(mins)

    # # print(list(*arr_mins))
    # plt.plot(freq)
    # plt.plot(smoothed)
    # plt.plot(arr_mins, smoothed[arr_mins], "x")
    # plt.show()

    # x1 = 0
    # lines = []

    # for x2 in list(*arr_mins):
    #     lines.append((x1,x2))
    #     x1=x2

    # lines.append((x1,img.shape[1]))

    lines = hard_seggregate(freq, 10)

    return lines


def get_words_x_coordinates(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(
        gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 1))

    # Appplying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    freq = [sum(column) // 255 for column in zip(*dilation)]

    words = hard_seggregate(freq, 1)

    return words

def resize_and_pad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

def get_characters(img):
    img = cv2.resize(img,None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
    img2 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    ret, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    eroded = cv2.erode(thresh1, kernel)
    freq = np.array([sum(column) // 255 for column in zip(*eroded)])

    characters = hard_seggregate(freq, 1)

    character_images=[]
    characters[0]
    for character in characters:
        character_image = thresh2[0:img.shape[0], character[0]:character[1]]
        character_image_64x64 = resize_and_pad(character_image, (64, 64), 255)
        # print(character_image_64x64)
        # cv2.imshow("r", character_image_64x64)
        # cv2.waitKey()
        character_images.append(character_image)

        cv2.rectangle(img2, (character[0], 0), (character[1], img.shape[0]), (0,0,255), 2)
        # cv2.line(img2, (character[1], 0), (character[1], img.shape[0]), (0,0,255), 2)
    
    
    plt.subplot(2, 2, 1), plt.imshow(eroded), plt.title("Eroded Image")
    plt.subplot(2, 2, 2), plt.imshow(thresh2), plt.title("Original Binary Image")
    plt.subplot(2, 2, 3), plt.plot(freq), plt.title("Frequency")
    plt.subplot(2, 2, 4), plt.imshow(img2), plt.title("Segmentation")
    plt.show()

    # for character in character_images:
    #     cv2.imshow("g", character)
    #     cv2.waitKey()

    return character_images


# reads an input image
img = cv2.imread('s2.png')
# img = cv2.resize(img,None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)

lines = get_lines_y_coordinates(img)
print("Lines:", lines)

img2 = img.copy()

for line in lines:
    line_start, line_end = line 
    line_img = img[line_start:line_end, 0:img.shape[1]]
    cv2.rectangle(img2, (0, line_start), (img.shape[1], line_end), (0, 0, 255), 1)

    words = get_words_x_coordinates(line_img)

    for word in words:
        word_start, word_end = word
        word_img = img[line_start:line_end, word_start:word_end]
        character_images = get_characters(word_img)

        word_string = ""

        for character_image in character_images:
            character = pytesseract.image_to_string(character_image)
            word_string += character

        print(word_string)

        cv2.rectangle(img2, (word_start, line_start), (word_end, line_end), (0, 255, 0), 1)

# cv2.imwrite('fd.jpg', img2)
# cv2.imshow("dhf", img2)
# cv2.waitKey()
