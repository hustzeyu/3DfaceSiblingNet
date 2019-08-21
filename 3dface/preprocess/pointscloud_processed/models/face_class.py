#coding:UTF-8
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import cv2
sys.path.insert(0, "/home/gp/caffe/python")
import caffe
import os
import time
import copy


class FaceDetect(object):
    """
    get face_box from a jpg. 
    Return [] if not face_box in a jpg.
    """
    def __init__(self, gpu=0):
        caffe.set_mode_gpu()
        caffe.set_device(gpu)
        self.model_def = "/home/gp/work/project/3d_face/datas/get_data_tool/face_det/code/models/face_detect/deploy.prototxt"
        self.model_weights = "/home/gp/work/project/3d_face/datas/get_data_tool/face_det/code/models/face_detect/SFD.caffemodel"
        self.net = caffe.Net(self.model_def, self.model_weights, caffe.TEST)

    def getboxes(self, image):
        """
        get list of face_boxes.
        face_box: [xmin, ymin, xmax, ymax]

        """
        heigh = image.shape[0]
        width = image.shape[1]
        im_shrink = 640.0 / max(image.shape[0], image.shape[1])
        image = cv2.resize(image, None, None, fx=im_shrink, fy=im_shrink, interpolation=cv2.INTER_LINEAR)
        hs = image.shape[0]
        ws = image.shape[1]
        tempimg = np.zeros((1, hs, ws, 3))
        tempimg[0, :, :, :] = image
        tempimg = tempimg.transpose(0, 3, 1, 2)
        self.net.blobs['data'].reshape(1, 3, hs, ws)
        self.net.blobs['data'].data[...] = tempimg
        self.net.forward()
        detections = copy.deepcopy(self.net.blobs['detection_out'].data[...])
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]
        faceboxes = []
        for i in xrange(det_conf.shape[0]):
            score = det_conf[i]
            if score > 0.8:
                rect = []
                xmin = int(round(det_xmin[i] * width))
                ymin = int(round(det_ymin[i] * heigh))
                xmax = int(round(det_xmax[i] * width))
                ymax = int(round(det_ymax[i] * heigh))
                half_rect_width = (xmax - xmin) * 0.5
                half_rect_heigh = (ymax - ymin) * 0.5
                xmin = xmin - half_rect_width
                ymin = ymin - half_rect_heigh
                xmax = xmax + half_rect_width
                ymax = ymax + half_rect_heigh
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(xmax, width - 1)
                ymax = min(ymax, heigh-1)
                rect.append(xmin)
                rect.append(ymin)
                rect.append(xmax)
                rect.append(ymax)
                # filter small face_box
                if ymax-ymin < heigh*0.25 or xmax-xmin < width*0.25:
                    continue
                faceboxes.append([int(i) for i in rect])
        return faceboxes

    def get_big_box(self, faceboxes):
        """
        get the biggest facebox from faceboxes.
        """
        facebox = faceboxes[0]
        area = (facebox[2]-facebox[0]) * (facebox[3]-facebox[1])
        for box in faceboxes[1:]:
            area_new = (box[2]-box[0]) * (box[3]-box[1])
            if area_new > area:
                facebox = box
                area = area_new
        return facebox

    def shrink_box(self, box, ratio):
        """
        shrink facebox by the ratio.
        """
        w = box[2]-box[0]
        h = box[3]-box[1]
        xmin = int(box[0] + ratio*w)
        ymin = int(box[1] + ratio*h)
        xmax = int(box[2] - ratio*w)
        ymax = int(box[3] - ratio*h)
        return [xmin, ymin, xmax, ymax]


class FaceLandmarks(object):
    """
    get face_landmarks from a img_face.
    """
    def __init__(self, gpu=0):
        caffe.set_mode_gpu()
        caffe.set_device(gpu)
        self.model_def = "/home/gp/work/project/3d_face/datas/get_data_tool/face_det/code/models/face_landmark/faceunet.prototxt"
        self.model_weights = "/home/gp/work/project/3d_face/datas/get_data_tool/face_det/code/models/face_landmark/u_landmark_pose.caffemodel"
        self.net = caffe.Net(self.model_def, self.model_weights, caffe.TEST)

    def get_points(self, img):
        half_width = img.shape[1] * 0.5
        half_height = img.shape[0] * 0.5
        tempimg = np.zeros((1, 48, 48, 3))
        scale_img = cv2.resize(img,(48,48))
        #scale_img = (scale_img - 127.5) / 125.0
        scale_img = (scale_img - 127.5)  * 0.0078125
        tempimg[0, :, :, :] = scale_img
        tempimg = tempimg.transpose(0, 3, 1, 2)
        self.net.blobs['data'].data[...] = tempimg
        self.net.forward()
        points = copy.deepcopy(self.net.blobs['fc2'].data[0])
        pose = copy.deepcopy(self.net.blobs['fc3'].data[0])
        #print "points: ", points
        #print "pose: ", pose / np.pi * 180
        #points = copy.deepcopy(self.net.blobs['conv6_2'].data[0])
        #pose = copy.deepcopy(self.net.blobs['conv6_1'].data[0])
        newlandmark = []
        facelandmarks = []
        for i in range(5):
            x = points[i * 2 + 0] * half_width + half_width
            y = points[i * 2 + 1] * half_height + half_height
            point = []
            point.append(int(x))
            point.append(int(y))
            facelandmarks.append(point)
            #if i == 7 or i == 10 or i == 14 or i == 17 or i == 19:
            #    newlandmark.append(point)
        #pose = pose / np.pi * 180
        #return facelandmarks, pose
        return facelandmarks

    def draw_point(self, img, face_box, facelandmarks):
        x, y, x1, y1 = face_box
        cv2.rectangle(img, (x,y), (x1,y1), (0,0,255), 2)
        face_img = img[y:y1, x:x1]
        for (x0, y0) in facelandmarks:
            x0 += x
            y0 += y
            cv2.circle(img, (x0,y0), 1, (0,0,255), 5)
        return img
        
    def alignment(self, input_img, facelandmarks):
        """
        align the input_img to 112x112.
        """
        points = []
        for (x0,y0) in facelandmarks:
            points.append(x0)
            points.append(y0)

        src = np.matrix([[points[0], points[2], points[4], points[6], points[8]],
                         [points[1], points[3], points[5], points[7], points[9]],
                         [1, 1, 1, 1, 1]])
        dst = np.matrix([[30.2946, 65.5318, 48.0252, 33.5493, 62.7299],
                         [46.6963, 46.5014, 66.7366, 87.3655, 87.2041]])
        T = (src * src.T).I * src * dst.T
        img_affine = cv2.warpAffine(input_img, T.T, (112, 112))
        return img_affine

def crop_img(img, box):
    x,y,x1,y1 = box
    return img[y:y1, x:x1]

if __name__ == "__main__":
    img = cv2.imread("/home/gp/work/project/3d_face/datas/get_data_tool/face_det/dataset/ceeeae5c-7e0f-11e9-8443-2c4d54e8b3b6/0-ceeeae5c-7e0f-11e9-8443-2c4d54e8b3b6.jpg")
    face_detect = FaceDetect()
    face_landmarks = FaceLandmarks()
    faceboxes = face_detect.getboxes(img)
    print faceboxes
    for box in faceboxes:
        face_img = crop_img(img, box)
        points = face_landmarks.get_points(face_img)
        img = face_landmarks.draw_point(img, box, points)

    cv2.namedWindow('Test Example', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Test Example', img)
    key = cv2.waitKey(10000)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
