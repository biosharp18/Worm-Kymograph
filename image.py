import numpy as np
import cv2
import math
import time
from scipy.interpolate import UnivariateSpline, interp1d
from scipy import interpolate
from scipy import optimize
from scipy import integrate
from scipy import signal
import torch



def ChooseLargestBlob(image):

    '''
    Given a binary image return an image that contains only the largest connected component
    Input:
        image (np.array): binary image of CNN output.

    Output:
        np.array: binary image with largest component
    '''
    connectivity = 4  #4-connectivity
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    t = np.zeros(image.shape)
    t[output == np.argmax(sizes)+1] = 255
    t = t.astype(np.uint8)
    return t

def fillHoles(outline):
    """
    Fill in holes (if any) of segmented outline of worm

    Input:
        outline (np.array): binary image of segmented worm outline.

    Output:
        np.array: Image of worm outline with holes filled.
    """
    orig_outline = np.copy(outline)
    h, w = outline.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(outline, mask, (0,0), 255)
    outline = cv2.bitwise_not(outline)
    outline_holes_filled = cv2.bitwise_xor(outline, orig_outline)
    return outline_holes_filled

def skeletonize(filled_outline):
    """
    Skeletonize the segmented outline of the worm in image

    Input:
        filled_outline (np.array): binary image of worm outline with no holes.

    Output:
        np.array: Image of worm skeleton (i.e centreline that fits the outline) with borders set to black.

    """
    outline = cv2.cvtColor(filled_outline,cv2.COLOR_GRAY2RGB)
    skeleton = cv2.ximgproc.thinning(cv2.cvtColor(outline, cv2.COLOR_RGB2GRAY))
    skeleton[-1] = 0 # set borders to black
    skeleton[0] = 0
    skeleton[:,-1] = 0
    skeleton[:,0] = 0
    return skeleton

def erode(image, erode_param = 21):
    """
    Performs morphological erosion on worm segmented mask

    Input:
        image (np.array): binary image of worm segmentation
        erode_param (odd int): size of erosion kernel

    Output:
        np.array: Image of worm skeleton,eroded
    """
    kernel = np.ones((erode_param,erode_param), np.uint8)
    return cv2.erode(image, kernel)

#def segment()


def sortSkeleton(skeleton_img):
    """
    Sorts the pixels of a skeletonized image from tail (pixel closest to border) to head. Pixels are ordered based on distance from e/o

    Input:
        skeleton_img (np.array): binary image of worm skeleton
    Output:
        np.array: List of pixels ordered from head to tail.
    """
    points = np.column_stack(np.where(skeleton_img != 0)) #get all points of skeleton
    #self.points = points
    img_height = skeleton_img.shape[0]
    img_width = skeleton_img.shape[1]
    y_s = points[:,0]
    x_s = points[:,1]
    ymax = y_s.max()
    ymaxarg = y_s.argmax()

    ymin = y_s.min()
    yminarg = y_s.argmin()

    xmax = x_s.max()
    xmaxarg = x_s.argmax()

    xmin = x_s.min()
    xminarg = x_s.argmin()


    if xmax >= img_width - 2: #touches right border 590

        start_point = points[xmaxarg,:]
        #print("right")
    if xmin <= 2: #touches left border, primary dims

        #print(xminarg)
        start_point = points[xminarg,:]
        #print("left")
        #print(start_point)
    if ymax >= img_height - 2:

        start_point = points[ymaxarg,:]
        #print("bottom")
    if ymin <= 2:

        start_point = points[yminarg,:]
        #print("top")
    #For each point, calculate distance of every point. Select lowest distance as next point, append to list, then delete from points
    order_list = np.empty((0,2))
    order_list = np.append(order_list, [start_point], axis=0)
    #print(order_list)
    not_yet_travelled = points
    #print(not_yet_travelled)
    not_yet_travelled = np.delete(not_yet_travelled, np.where((not_yet_travelled == start_point).all(axis=1))[0],0) #delete first point since already travelled
    cur_point = start_point #we start at the start

    #print(points)
    #print(not_yet_travelled)
    while True: #keep on looping
        min_dist = 1000000
        if len(not_yet_travelled) == 0:
            break
        for next_point in not_yet_travelled: #search
            distance = math.dist(cur_point, next_point)
            if distance < min_dist:
                min_dist = distance
                closest_point = next_point

        #print(closest_point)
        #after for loop we should have the closest point be the actual closest point. Append the closest point to order list
        not_yet_travelled = np.delete(not_yet_travelled, np.where((not_yet_travelled == closest_point).all(axis=1))[0],0)
        order_list = np.append(order_list, [closest_point], axis=0)
        cur_point = closest_point

    return np.flipud(order_list)
    #return x_s, y_s, points

def padEdges(image):
    """
    Concatenates zeros to image width and height, namely to prevent indexing errors
    """
    #img_width = image.shape[0]
    #img_height = image.shape[1]
    image = np.concatenate((image, np.zeros((30, image.shape[1]))),axis=0)
    image = np.concatenate((image, np.zeros((image.shape[0],30 ))),axis=1)
    return image


def normalizeImage(raw_img):
    return 255.0/raw_img.max() * raw_img

def getPrediction(image, model, device):
    image = normalizeImage(image)
    original_dim = image.shape
    #skeletonization params #here is the issue, original arr and arr
    #print("original mean", np.mean(original_arr))
    raw_img = cv2.resize(image, (256,256))
    #print("resized mean",np.mean(arr))
    raw_img = raw_img/255
    raw_img = np.expand_dims(raw_img, 0)
    img_tensor = torch.from_numpy(raw_img).float().to(device).unsqueeze(0)
    #print(img_tensor)
    pred = model(img_tensor)
    output = pred.detach().cpu().numpy().squeeze() * 255
    #print(type(output))
    #print(np.flip(original_dim))
    output = cv2.resize(output, tuple(np.flip(original_dim)))
    output = cv2.threshold(output, 0,255, cv2.THRESH_BINARY)[1]
    return output.astype(np.uint8)