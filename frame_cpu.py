import torch
from torch import nn
import os
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import time
from scipy.interpolate import UnivariateSpline, interp1d
from scipy import interpolate
from scipy import optimize
from scipy import integrate
from scipy import signal
#from arc_length_1 import *
from scipy import stats as st

os.chdir("C:/Users/roryg/Desktop/Zhen Lab/C.Elegans_video_analysis")
from image import *
from process import *
from classes import *
#from google.colab.patches import cv2_imshow

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cpu")
filename = 'h5_20220922_2 1774.h5'




##for local implementation
os.chdir('D:\\Models')
segnet = SegNet()
segnet.load_state_dict(torch.load("ver15.pth",map_location=device))
segnet.eval()



#os.chdir('..')
os.chdir("D:\\Datasets\\Bad alignment")
f = h5py.File(filename, 'r') # Load h5
spacing = 3 #Paramter for number of sub-pixel body points on kymograph. 1 = 1 point per pixel
total_frames = len(f.keys()) - 1


arr = np.zeros(np.shape(f['t00000']['s00']['0']['cells'])[1:])
j = '00000'
for i in range(np.shape(f['t00000']['s00']['0']['cells'])[0]): #Sum up z stack
  t = f['t'+ j]['s00']['0']['cells']
  arr = arr + t[i] # arr holds the summed up z stack image
arr = arr.T #for test only#start_time = time.time()
original_dim = arr.shape
original_arr = np.array(arr)





##For first frame
#original_img = normalizeImage(original_arr) #skeletonization params #here is the issue, original arr and arr
  #print("original mean", np.mean(original_arr))
mask = getPrediction(original_arr, segnet, torch.device("cpu"))



mask = ChooseLargestBlob(mask)
mask_filled = fillHoles(mask)
mask_erode = erode(mask_filled)
mask_skeleton = skeletonize(mask_erode)

body_points = sortSkeleton(mask_skeleton)
original_img = padEdges(original_arr)

[spline_x, spline_y, t_vals, x_vals, y_vals] = fitSpline(body_points)


global first_intestine_ind

[total_val, xpoints, cur_arc_length, first_intestine_ind] = integrateImage(original_img, spline_x, spline_y, t_vals, x_vals, y_vals)

total_valf = total_val.copy()
xpointsf = xpoints.copy()

global max_arc_length
max_arc_length = math.floor(0.9*xpoints.max()) #first frame sets the kymograph y dim for all frames.
global time_series
time_series = np.zeros([total_frames, max_arc_length*spacing+400])




waveform = InterpolateIntensity(total_val, xpoints, max_arc_length, spacing) #ONLY FOR FIRST ONE NOW

kymo_length = max_arc_length *spacing
        #plotting just entry is good enough


time_series[0] = waveform


frame.kymo_length = kymo_length

for j in range(2,3): #Loop over how many frames you want to analyse
    k = j
    arr = np.zeros(np.shape(f['t00000']['s00']['0']['cells'])[1:])
    j = str(j)
    lenj = len(j)
    for instance in range(5-lenj):
        j = '0' + j
    #print(j)
    for i in range(np.shape(f['t00000']['s00']['0']['cells'])[0]): #Sum up z stack
        t = f['t'+ j]['s00']['0']['cells']

        arr = arr + t[i] # arr holds the summed up z stack image
        #print(arr)

    arr = arr.T #for test only

    start_time = time.time()
    original_dim = arr.shape
    original_arr = np.array(arr)


#print(arr*30)

    #plt.imshow(arr.squeeze())


  #cv2_imshow(segnet(img_tensor).detach().cpu().numpy().squeeze()*255)'



    frame = Frame(original_arr, 21) #skeletonization params #here is the issue, original arr and arr

    #print("original mean", np.mean(original_arr))
    raw_img = cv2.resize(frame.raw, (256,256))
    #print("resized mean",np.mean(arr))
    raw_img = raw_img/255
    raw_img = np.expand_dims(raw_img, 0)
    img_tensor = torch.from_numpy(raw_img).float().to(device).unsqueeze(0)
    #print(img_tensor)
    pred = segnet(img_tensor)
    output = pred.detach().cpu().numpy().squeeze() * 255
    #print(type(output))
    #print(np.flip(original_dim))
    output = cv2.resize(output, tuple(np.flip(original_dim)))
    output = cv2.threshold(output, 0,255, cv2.THRESH_BINARY)[1]



    frame.outline = output.astype(np.uint8)
    frame.outline = ChooseLargestBlob(frame.outline)
    frame.fillHoles()

    frame.erode()
    frame.skeletonize()

    frame.kymo_length = kymo_length

    body_points = frame.sort_skeleton()
    frame.raw = np.concatenate((frame.raw, np.zeros((30, frame.raw.shape[1]))),axis=0)
    frame.raw = np.concatenate((frame.raw, np.zeros((frame.raw.shape[0],30 ))),axis=1)
    frame.spline_fit(body_points)
    print("ok")

    #frame.prev_wave = widthform

    #[total_val, xpoints, cur_arc_length, lag, widthform] = frame.vector(save=True)
    [total_val, xpoints, cur_arc_length, intestine_ind] = frame.vector(save=True)
    #xpoints = xpoints + lag\
    xpoints = xpoints- (intestine_ind - first_intestine_ind)
    frame.InterpolateIntensity(total_val, xpoints, max_arc_length, spacing)
    #plt.plot(xpoints,total_val/15)
    plt.plot(frame.waveform)
    #plt.xlim(0,3000)
    #plt.savefig("b"+j +".png")
    #plt.close()

    #


    time_series[k] = frame.waveform
    secs = time.gmtime((total_frames - k)*(time.time()-start_time))
    time_remaining = time.strftime("%H:%M:%S", secs)
    print("Done t=" + str(k) + " out of "+ str(total_frames) + "Estimated time remaining:" + str(time_remaining))

##Display kymograph with color bar
time_series = time_series.T
#fig, ax = plt.subplots()
#cax = ax.imshow(time_series, cmap=cm.viridis)
#minp = time_series.min()
#maxp = time_series.max()
#cbar = fig.colorbar(cax, ticks=[minp,(minp+maxp)/2,maxp])
#cbar.ax.set_yticklabels([str(minp),str((minp+maxp)/2),str(maxp)])
#plt.show()
f.close()

#time_series = RealignKymo(time_series, intestine_start_pos+200, intestine_max_pos+200)

#np.save(str(filename)+".npy", time_series)

#plt.plot(time_series[:,0])
#print("Identify start of intestine:")
#plt.show() #identify start of grinder
#global intestine_start_pos
#intestine_start_pos = int(input())
#print("Identify max postion of intestine")
#global intestine_max_pos
#intestine_max_pos = int(input())


#newkym = RealignKymo(time_series, intestine_start_pos, intestine_max_pos)

