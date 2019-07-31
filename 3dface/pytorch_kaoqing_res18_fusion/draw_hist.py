#coding:UTF-8
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import os
import numpy as np
import cv2
import os
from tqdm import tqdm
import time
import itertools
import random
from test import Feature_extract
import matplotlib.pyplot as plt
from config import Config
from torch.nn import DataParallel


def Score_analyse(same_score, diff_score, name):
    plt.title("same/different pairs seq score histogram")
    plt.hist(same_score, bins=100, normed = 1, facecolor="red", edgecolor="black", label='same', alpha=0.7, hold = 1)
    plt.hist(diff_score, bins=100, normed = 1, facecolor="blue", edgecolor="black",label='different', alpha=0.7)
    plt.xlabel("score")
    plt.ylabel("number")
    plt.legend()
    plt.savefig(name)
    plt.show()

def get_img_list(test_list):
    with open(test_list, "r") as f:
        lines = f.readlines()
    img_list = [i.split(" ")[0] for i in lines]
    return img_list

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
            if filename.split('.')[-1] in suffix and "finally" in filename:
                imglist.append(filepath)
    return imglist


def cos_sim(vector_a, vector_b):
    """
    compute the cos_sim between two vectors
    param: vector_a
    param: vector_b
    return: cos_sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5*cos
    return sim


def get_pairs(img_list):
    """
    返回可迭代对象，里面是每对图片的路径。
    """
    return itertools.combinations(img_list, 2)
 

if __name__ == "__main__":
    opt = Config()
    extract = Feature_extract()
    test_list = "/home/gp/work/project/kaoqing/img_list/test_color_depth.list"
    img_list = get_img_list(test_list)
    #test_dir = "/home/gp/work/project/3d_face/datas/public_data/data/" 
    #img_list = get_image_list(test_dir)
    pairs = get_pairs(img_list)
    pairs = [i for i in pairs]
    random.shuffle(pairs)
    print len(list(pairs))
    same_list, diff_list = [], []
    same_score, diff_score = [], []
    for pair in tqdm(pairs):
        if pair[0].split("/")[-2] == pair[1].split("/")[-2]:
            flag = 1
            line = "%s %s\n" % pair
            if len(same_list) < 6000:
                same_list.append(line)
                feature0 = extract.feature_extract(pair[0])
                feature1 = extract.feature_extract(pair[1])
                sim = cos_sim(feature0, feature1)
                same_score.append(sim)
            if len(same_list) == 6000:
                break
        else:
            flag = 0
            line = "%s %s\n" % pair
            if len(diff_list) < 6000:
                diff_list.append(line)
                feature0 = extract.feature_extract(pair[0])
                feature1 = extract.feature_extract(pair[1])
                sim = cos_sim(feature0, feature1)
                diff_score.append(sim)
    Score_analyse(same_score, diff_score, "test.png")
