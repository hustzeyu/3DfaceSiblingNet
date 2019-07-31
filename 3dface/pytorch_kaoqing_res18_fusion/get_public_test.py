# coding: UTF-8
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import numpy as np
import cv2
from tqdm import tqdm


def get_image_list(image_dir, suffix=['jpg', 'jpeg', 'JPG', 'JPEG','png']):
    '''get all image path ends with suffix'''
    if not os.path.exists(image_dir):
        print "PATH:%s not exists" % image_dir
        return []
    imglist = []
    for root, sdirs, files in os.walk(image_dir):
        if not files:
            continue
        for filename in files:
            filepath = os.path.join(root, filename)
            if filename.split('.')[-1] in suffix and "color" in filename:
                imglist.append(filepath)
    return imglist



if __name__ == "__main__":
    img_dir = "/home/gp/work/project/3d_face/datas/public_data/data/"
    img_list = get_image_list(img_dir)
    for img_path in tqdm(img_list):
        color_img = cv2.imread(img_path)
        rst_img_path = img_path.replace("color", "rst")
        rst_img = cv2.imread(rst_img_path)
        finall_img = np.hstack([rst_img, color_img])
        save_path = img_path.replace("color", "finally")
        cv2.imwrite(save_path, finall_img)

    

