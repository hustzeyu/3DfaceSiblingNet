def get_result(net, img):

    half_width = img.shape[1] * 0.5
    half_height = img.shape[0] * 0.5

    tempimg = np.zeros((1, 48, 48, 3))
    scale_img = cv2.resize(img,(48,48))
    #scale_img = (scale_img - 127.5) / 125.0
    scale_img = (scale_img - 127.5)  * 0.0078125
    tempimg[0, :, :, :] = scale_img

    tempimg = tempimg.transpose(0, 3, 1, 2)
    net.blobs['data'].data[...] = tempimg

    net.forward()
    points = copy.deepcopy(net.blobs['fc2'].data[0])
    pose = copy.deepcopy(net.blobs['fc3'].data[0])

    print "points: ", points
    print "pose: ", pose / np.pi * 180

    #points = copy.deepcopy(net.blobs['conv6_2'].data[0])
    #pose = copy.deepcopy(net.blobs['conv6_1'].data[0])

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
