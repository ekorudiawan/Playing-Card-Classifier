import numpy as np 
import cv2 as cv
import os 
import glob
import random
import string

ori_img_path = "/home/Playing-Card-Classifier/original_images/"
dataset_path = "/home/Playing-Card-Classifier/dataset/"
label_path = "/home/Playing-Card-Classifier/dataset/label/"
n_synthetic_img = 5

def random_string(length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def change_brightness(img, brightness=50):
    return np.clip(1.0 * img + brightness, 0, 255).astype(np.uint8)

def change_size(img, scale=1.0):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    return cv.resize(img, dim, interpolation = cv.INTER_AREA)

def rotate_image(img, angle=0):
    rows, cols, _ = img.shape
    M = cv.getRotationMatrix2D((cols/2,rows/2),angle,1)
    return cv.warpAffine(img,M,(cols,rows))

def put_rectangle(img, rect_w=100, rect_h=200, pos=(200,200)):
    pos_x, pos_y = pos
    top_left_x = pos_x - (rect_w//2)
    top_left_y = pos_y - (rect_h//2)
    bottom_right_x = pos_x + (rect_w//2)
    bottom_right_y = pos_y + (rect_h//2)
    return cv.rectangle(img,(top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0,0,0), -1)

def main():
    print("Image Augmentation Tool")
    label_file = open(label_path+"labels.txt","w+") 
    files = [f for f in glob.glob(ori_img_path + "*.jpg", recursive=True)]
    for f in files:
        f = f.replace(ori_img_path,'')
        f = f.replace('.jpg','')
        label = f 
        
        for i in range(n_synthetic_img):
            img = cv.imread(ori_img_path+f+".jpg")
            h, w, _ = img.shape
            random_filename = random_string()
            if i == 0:
                cv.imwrite(dataset_path+random_filename+".jpg", img)
                label_file.write(random_filename+","+label+"\r\n")

            brightness_val = random.randint(-10, 10)
            random_filename = random_string()
            b_img = change_brightness(img, brightness=brightness_val)
            cv.imwrite(dataset_path+random_filename+".jpg", b_img)
            label_file.write(random_filename+","+label+"\r\n")

            scaling_val = random.uniform(0.9, 1.1)
            rz_img = change_size(img, scale=scaling_val)
            random_filename = random_string()
            cv.imwrite(dataset_path+random_filename+".jpg", rz_img)
            label_file.write(random_filename+","+label+"\r\n")

            rot_val = random.randint(-10, 10)
            rot_img = rotate_image(img, angle=rot_val)
            random_filename = random_string()
            cv.imwrite(dataset_path+random_filename+".jpg", rot_img)
            label_file.write(random_filename+","+label+"\r\n")

            pos = (random.randint(0, w), random.randint(0, h))
            rect_w = random.randint(0, 25*w//100)
            rect_h = random.randint(0, 25*h//100)
            rect_img = put_rectangle(img, rect_w=rect_w, rect_h=rect_h, pos=pos)
            random_filename = random_string()
            cv.imwrite(dataset_path+random_filename+".jpg", rect_img)
            label_file.write(random_filename+","+label+"\r\n")
    print("Image Augmentation Finish")
if __name__ == "__main__":
    main()