#coding:UTF-8
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import numpy as np
import cv2
import random


def get_image_list(image_dir, suffix=['jpg', "png"]):
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
            if filename.split('.')[-1] in suffix:
                imglist.append(filepath)
    return imglist

def split(imglist, train_file, test_file):
    split_point = int(len(img_list)/100)
    with open(train_file, "w") as f1:
        f1.writelines(imglist[:-split_point])
    with open(test_file, "w") as f2:
        f2.writelines(imglist[-split_point:])

if __name__ == "__main__":
    indir1 = "/home/hzy/Documents/work/preprocess/try/rgb_112"
    #indir2 = "/home/gp/work/project/3d_face/datas/public_data/data/WRL31-60"
    #indir3 = "/home/gp/work/project/3d_face/datas/public_data/data/WRL61-90"
    #indir4 = "/home/gp/work/project/3d_face/datas/public_data/data/WRL91-110"
    img_list = []
    for indir in [indir1]:
        imglist = get_image_list(indir)
        img_list += imglist
    img_list = [i+" "+str(int(i.split("/")[-2].split("_")[-1])-1) + "\n" for i in img_list]
    random.shuffle(img_list)
    split(img_list, "train_rgb.list", "test_rgb.list")
