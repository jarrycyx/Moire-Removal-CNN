import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image
import PIL
import os

num = 0

def rotate(img, rot_angle):
    (h, w) = img.shape[:2] 
    center = (w // 2, h // 2) 
    M = cv2.getRotationMatrix2D(center, rot_angle, 1.0) 
    rotated = cv2.warpAffine(img, M, (w, h))
    rotated_cropped = rotated[int(rotated.shape[0]*0.147):int(rotated.shape[0]*0.854), int(rotated.shape[1]*0.147):int(rotated.shape[1]*0.854), :]
    return rotated_cropped

def generate_img_with_moire(original_img_path,
                            res = 1000,
                            moire_prop = 0.1,
                            moire_cycle = 10,
                            rot_angle = 30,
                            down_sampling_mode = 10.7):
    
    global num
    num += 1

    img_mask = np.ones([res, res, 3])
    img_mask = [img_mask[index]*(moire_prop*np.sin(2*np.pi/moire_cycle*index)+1-moire_prop) for index in range(res)]
    # print(img_mask)

    img_content = np.ones([res, res, 3])
    #img_content[int(res/4):int(res/4*3), int(res/4):int(res/4*3), :] = img_content[int(res/4):int(res/4*3), int(res/4):int(res/4*3), :] - np.ones([int(res/2), int(res/2), 3])/2

    img_original = cv2.imread(original_img_path) / 255
    img_content = cv2.resize(img_original, (res, res), interpolation = cv2.INTER_CUBIC)
    img_content[:, :, [0, 1, 2]] = img_content[:, :, [2, 1, 0]]

    img = img_mask * img_content


    img_downsampling = np.zeros([int(res/down_sampling_mode), int(res/down_sampling_mode), 3])
    for x in range(img_downsampling.shape[0]):
        for y in range(img_downsampling.shape[0]):
            img_downsampling[x, y, :] = img[int(x*down_sampling_mode), int(y*down_sampling_mode), :]

    cv2.imwrite(str(num) + "original.jpg", rotate(img_content, rot_angle)*255)
    cv2.imwrite(str(num) + "moire.jpg", rotate(img_downsampling, rot_angle)*255)




dir_path = "D:/11PRojects/ML_DL/moire_dataset/original_img/"
img_list = os.listdir(dir_path)
for path in img_list:
    for i in range(5):
        this_moire_cycle = (np.random.rand(1)*5+3)[0]
        this_rot_angle = (np.random.rand(1)*180)[0]
        this_down_sampling_mode = this_moire_cycle+np.random.rand(1)[0]*3+0.5
        generate_img_with_moire(dir_path + path, moire_cycle=this_moire_cycle, rot_angle=this_rot_angle, down_sampling_mode=this_down_sampling_mode)
    