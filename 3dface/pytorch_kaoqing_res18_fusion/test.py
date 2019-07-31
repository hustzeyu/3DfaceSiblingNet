# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
from __future__ import print_function
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import cv2
from models import resnet, metrics, focal_loss
import torch
import numpy as np
import time
import itertools
import copy
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
from config import Config
from torch.nn import DataParallel
from tqdm import tqdm

opt = Config()

class Feature_extract(object):
    def __init__(self):
        self.device = torch.device("cuda")
        self.model = resnet.resnet_face18(opt.use_se)
        self.model = DataParallel(self.model)
        self.model.load_state_dict(torch.load(opt.test_model_path))
        self.model.to(self.device)

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.transforms = T.Compose([
                     T.ToTensor(),
                     normalize
                     ])
   
    def feature_extract(self, img_path):
        img = Image.open(img_path)
        img = img.resize((112, 112))
        img = self.transforms(img)
        img = img.unsqueeze(0)
        with torch.no_grad():
            self.model.eval()
            data_input = img.to(self.device)
            feature = self.model(data_input)
        feature = np.array(feature.cpu())[0, :].tolist()
        vector = np.mat(feature)
        denom = np.linalg.norm(vector)
        return (np.array(feature) / denom).tolist()


def SimilarityMatrixCal(listFeature,listID,resultTxt):
    steps = 100
    step = 1.0 / steps
    fpr_TH_001 =0.001
    fpr_TH_0001 =0.0001

    re_fpr, re_tpr, re_acc = [], [], []
    f_re = open(resultTxt, 'w')
    arrFea = np.array(listFeature)
    arrFea_T = arrFea.T
    SimilarityMatrix = np.dot(arrFea , arrFea_T)
    LabelMatrix = SimilarityMatrix * 0
    Num = LabelMatrix.shape[0]

    sameList = []
    diffList = []
    for i in range(Num):
        for j in range(i+1, Num):
            if listID[i] == listID[j]:
                sameList.append(SimilarityMatrix[i,j])

            else:
                diffList.append(SimilarityMatrix[i,j])

    sameNum = len(sameList)
    diffNum = len(diffList)
    ratio = sameNum *1.0  / diffNum

    print('\n')
    reStr = "same pairs = " + str(sameNum) +" , diff pairs = " + str(diffNum)+" ,  ratio = " + str(ratio)
    print(reStr)
    f_re.writelines(reStr + '\n')
    sigma = 0
    s1 = np.array(sameList)
    s2 = np.array(diffList)
    for i in range(steps):
        sigma += step   # step = 0.01
        s1[s1 < sigma] = 0
        TPnum = np.count_nonzero(s1)
        FNnum = sameNum - TPnum
        s2[s2 < sigma] = 0
        FPnum = np.count_nonzero(s2)
        TNnum = diffNum - FPnum
        if TPnum + FNnum != 0 and FPnum + TNnum != 0:
            fpr = FPnum *1.0 / (TNnum + FPnum)
            tpr = TPnum *1.0 / (TPnum + FNnum)
            acc = (TPnum + TNnum * ratio) / ((TPnum + FNnum) + (TNnum + FPnum) * ratio)
            re_fpr.append(fpr)
            re_tpr.append(tpr)
            re_acc.append(acc)
            reStr = "sigma= " + str(round(sigma,2)) + " , fpr= " + str(fpr) + " , tpr= " + str(tpr) + " , acc= " + str(acc) + " , FP= " + str(FPnum) + " , TP= " + str(TPnum)
            print(reStr)
            f_re.writelines(reStr + '\n')
        else:
            print ("result file is wrong")
            exit(-1)
    final_001_fpr_index = np.argmin(np.abs(np.array(re_fpr)-fpr_TH_001))
    final_001_fpr = re_fpr[final_001_fpr_index]
    final_sigma_001 = final_001_fpr_index * step + step
    final_001_tpr = re_tpr[final_001_fpr_index]

    final_0001_fpr_index = np.argmin(np.abs(np.array(re_fpr)-fpr_TH_0001))
    final_0001_fpr = re_fpr[final_0001_fpr_index]
    final_sigma_0001 = final_0001_fpr_index * step + step
    final_0001_tpr = re_tpr[final_0001_fpr_index]

    final_zero_fpr_index = re_fpr.index(0,1)
    final_zero_sigma = final_zero_fpr_index * step + step
    final_zero_tpr = re_tpr[final_zero_fpr_index]

    print('\n')
    reStr = 'The fpr that you care : '+ str(final_001_fpr)+ '   and the corresponding sigma : '+str(final_sigma_001)+'   and the corresponding tpr : '+str(final_001_tpr) + '\n'+ \
            'The fpr that you care : ' + str(final_0001_fpr) + '   and the corresponding sigma : ' + str(final_sigma_0001) + '   and the corresponding tpr : ' + str(final_0001_tpr) + '\n'+ \
            'The fpr that you care : 0 ,  and the corresponding sigma : '+str(final_zero_sigma)+'   and the corresponding tpr : '+str(final_zero_tpr)
    f_re.writelines(reStr + '\n')

    print(reStr)
    f_re.close()
    return re_fpr, re_tpr


def draw_roc(re_fpr, re_tpr):
    plt.plot(re_fpr, re_tpr)
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.title("ROC")
    plt.savefig("roc.png")
    plt.show()


def get_img_list(test_list):
    with open(test_list, "r") as f:
        lines = f.readlines()
    img_list = [i.split(" ")[0] for i in lines]
    return img_list

def get_image_list(image_dir, suffix=['jpg', 'jpeg', 'JPG', 'JPEG','png']):
    '''get all image path ends with suffix'''
    if not os.path.exists(image_dir):
        #print "PATH:%s not exists" % image_dir
        return []
    imglist = []
    for root, sdirs, files in os.walk(image_dir):
        if not files:
            continue
        for filename in files:
            filepath = os.path.join(root, filename)
            if filename.split('.')[-1] in suffix and int(filepath.split("/")[-2]) < 10:
                imglist.append(filepath)
    return imglist



if __name__ == '__main__':
    opt = Config()
    extract = Feature_extract()
    test_list = "/home/gp/work/project/kaoqing/img_list/test_color_depth.list"
    img_list = get_img_list(test_list)
    #test_dir = "/home/gp/work/project/kaoqing/data_process/3d_image_test"
    #img_list = get_image_list(test_dir)
    listFeature, listID = [], []
    for img_path in tqdm(img_list):
        feature = extract.feature_extract(img_path)
        ID = int(img_path.split("/")[-2].split("_")[-1]) - 1
        #ID = int(img_path.split("/")[-2])
        listFeature.append(feature)
        listID.append(ID)
    resultTxt = "result.txt"
    re_fpr, re_tpr = SimilarityMatrixCal(listFeature, listID, resultTxt) 
    draw_roc(re_fpr, re_tpr)

