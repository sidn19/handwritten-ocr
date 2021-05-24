import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmin
from tensorflow.keras.models import load_model
from PIL import Image
import pytesseract
from skimage.morphology import skeletonize


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

    rect_kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    rect_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))

    #noise removal
    eroded = cv2.erode(thresh1, rect_kernel1)
    dilation = cv2.dilate(eroded, rect_kernel2, iterations=1)

    # plt.subplot(3,1,1), plt.imshow(thresh1)
    # plt.subplot(3,1,2), plt.imshow(eroded)
    # plt.subplot(3,1,3), plt.imshow(dilation)
    # plt.plot()

    # cv2.imshow("sfhs", dilation)
    # cv2.waitKey()
    

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
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    freq = smooth(np.array([sum(column) // 255 for column in zip(*dilation)]), 20)

    words = hard_seggregate(freq, 1)

    # plt.subplot(2, 1, 1), plt.imshow(dilation)
    # # freq = np.array([sum(column) // 255 for column in zip(*dilation)])
    # plt.subplot(2, 1, 2), plt.plot(freq)
    # plt.show()

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

def expand_to_square(img):
    rows, columns = img.shape

    if rows == columns:
        return img
    
    padding = abs(rows - columns) // 2
    p=np.ones((rows, padding) if rows > columns else (padding, columns), np.uint8) * 255
    return np.concatenate((p,img,p), axis=1 if rows > columns else 0)


def get_characters(img):
    img = cv2.resize(img,None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
    img2 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    ret, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,100))

    eroded = cv2.erode(thresh1, kernel1)
    eroded //= 255
    skeleton = skeletonize(eroded)
    # dilation = cv2.dilate(eroded, kernel2, iterations=1)

    freq = np.array([sum(column) for column in zip(*skeleton)])

    # plt.subplot(4, 1, 1), plt.imshow(eroded)
    # plt.subplot(4, 1, 2), plt.imshow(skeleton)
    # plt.subplot(4, 1, 3), plt.plot(smooth(freq))
    # plt.show()

    characters = hard_seggregate(smooth(freq), 1)

    character_images=[]
    # characters[0]
    for character in characters:
        character_image = thresh2[0:img.shape[0], character[0]:character[1]]
        y_start, y_end = 0, character_image.shape[0]
        x_start, x_end = 0, character_image.shape[1]

        cv2.rectangle(img2, (character[0], 0), (character[1], img.shape[0]), (255,0,0), 1)
        
        for r in character_image:
            if all(c == 255 for c in r):
                y_start += 1
            else:
                break
        
        for i in range(character_image.shape[0]-1, 0, -1):
            if all(c == 255 for c in character_image[i]):
                y_end -= 1
            else:
                break
        
        for c in zip(*character_image):
            if all(r == 255 for r in c):
                x_start += 1
            else:
                break

        for i in range(character_image.shape[1]-1, 0, -1):
            if all(r == 255 for r in character_image[:,i]):
                x_end -= 1
            else:
                break
    
        cropped_character_image = character_image[y_start:y_end, x_start:x_end]
        squared_character_image = expand_to_square(cropped_character_image)

        character_image_32x32 = resize_and_pad(squared_character_image, (32, 32), 255)
        horizontal_pad = np.ones((32, 16), np.uint8) * 255
        tmp=np.concatenate((horizontal_pad, character_image_32x32, horizontal_pad), axis=1)
        vertical_pad = np.ones((16, 64), np.uint8) * 255
        character_image_64x64=np.concatenate((vertical_pad, tmp, vertical_pad), axis=0)

        character_images.append(character_image_64x64)
    
    plt.subplot(4, 1, 1), plt.imshow(eroded)
    plt.subplot(4, 1, 2), plt.imshow(skeleton)
    plt.subplot(4, 1, 3), plt.plot(smooth(freq))
    plt.subplot(4, 1, 4), plt.imshow(img2)
    plt.show()

    return character_images

def pil_to_opencv(img):
    # mirror
    img = img.convert('RGB')
    opencv_img = np.array(img)
    return opencv_img[:, :, ::-1].copy()

def image_to_text(img):
    # img = cv2.imread(path)
    img = pil_to_opencv(img)
    line_blocks = get_lines_y_coordinates(img)

    img2 = img.copy()

    dataset = 'lowercase'

    model = load_model(dataset + '.model')

    letters = "abcdefghijklmnopqrstuvwxyz"

    lines = []

    for line_block in line_blocks:
        line_start, line_end = line_block 
        line_img = img[line_start:line_end, 0:img.shape[1]]
        cv2.rectangle(img2, (0, line_start), (img.shape[1], line_end), (0, 0, 255), 1)

        word_blocks = get_words_x_coordinates(line_img)

        line = ""

        for word_block in word_blocks:
            word_start, word_end = word_block
            word_img = img[line_start:line_end, word_start:word_end]
            character_images = get_characters(word_img)
            
            word_batch = []

            for character_image in character_images:
                character_image_3d = np.array([[[0]]*64]*64)

                for i in range(64):
                    for j in range(64):
                        character_image_3d[i][j][0] = character_image[i][j]
                
                word_batch.append(character_image_3d)
            
            if word_batch:            
                word_predict = model.predict_classes(np.array(word_batch))
                word_string = "".join([letters[predicted_class] for predicted_class in word_predict])
                # print(word_string,spell.correction(word_string))

                line += word_string + ' '

            cv2.rectangle(img2, (word_start, line_start), (word_end, line_end), (0, 255, 0), 1)
        
        lines.append(line)

    # cv2.imshow("Bounding Boxes", img2)
    # cv2.waitKey()

    return '\n'.join(lines)


# for line in image_to_text(Image.open('samples\\sample3.png')):
#     print(line)

# print(pytesseract.image_to_string(Image.open('samples\\sample2.png')))
