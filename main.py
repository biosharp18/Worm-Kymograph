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
from scipy import stats as st
import model
import functions


device = torch.device("cpu")
filename = 'h5_20220902_5 1.h5'




##for local implementation
os.chdir('D:\\Models')
segnet = model.SegNet()
segnet.load_state_dict(torch.load("ver15.pth",map_location=device))
segnet.eval()



#os.chdir('..')
os.chdir("D:\\Datasets\\Bad alignment")
f = h5py.File(filename, 'r') # Load h5
spacing = 3 #Paramter for number of sub-pixel body points on kymograph. 1 = 1 point per pixel
total_frames = len(f.keys()) - 1




class Frame: #blur = 51, thresh = 3, erode = 17

    def __init__(self, raw_img, erode_param):
        self.raw = 255.0/raw_img.max() * raw_img
        self.raw = self.raw.astype(np.uint8)
        self.dim = self.raw.shape
        self.background_val = 0
        #self.blur_img = None
        #self.thresh_img = None
        #self.size_filter_img = None
        self.outline = None
        self.segmented = None
        self.outline_filled = None
        self.skeleton = None
        #self.blur_param = blur_param
        #self.thresh_param = thresh_param
        self.erode_param = erode_param
        #for intensity grabber
        self.points = None
        self.waveform = None
        self.grinder_pos = None
        self.intestine_start_ind = None
        self.intestine_start_length = None
        self.intestine_max_ind = None
        self.intestine_max_length = None
        #self.spline_img = None
        self.spline_x = None
        self.spline_y = None
        self.spline_x_deriv = None
        self.spline_y_deriv = None
        self.t_vals = None
        self.x_vals = None
        self.y_vals = None
        self.prev_wave = None
        self.arc_length = None
        self.kymo_length = None
    def blur(self):
        self.blur_img = cv2.medianBlur(self.raw, self.blur_param)
    def thresh(self):
        thresh_img = cv2.adaptiveThreshold(self.blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 255,self.thresh_param) # was 2 before
        self.thresh_img = cv2.bitwise_not(thresh_img)
    def identifyComponents(self,image):
        connectivity = 4  #4-connectivity
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        return [nb_components, output, stats, centroids, sizes]
    def blobAverageFilter(self, nb_components, output, sizes):
        min_size = np.average(sizes) #this the biggest object
        img2 = np.zeros((output.shape))
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255
        img2 = img2.astype(np.uint8)
        self.size_filter_img = img2
    #call blur, thresh, identify components, averaged filter ssizes, components, averaged filter sizes again, connected components, noise filtering
    def blobNoiseFilter(self, nb_components, output, sizes):
        std_dev_list = []
        for i in range(0, nb_components): # for each component, we want the standard deviation of the pixel values of arr in it. That means we want a list of pixel values, we want a matrix of only component values.
            dark = np.zeros(self.dim)
            dark.fill(np.nan) #just so we can differentiate zeros of component etc...
            dark[output == i + 1] = self.raw[output == i + 1] #make the component part arr
            dark = dark.flatten()
            dark = dark[np.logical_not(np.isnan(dark))]
            std_dev = math.sqrt(np.mean(dark)) ##Just cuz it looks like a poisson distribution
            std_dev_list.append(std_dev)
        worm_n = std_dev_list.index(min(std_dev_list)) + 1
        self.outline = np.zeros(self.dim)
        self.outline[output == worm_n] = 255
        self.outline = self.outline.astype(np.uint8)
    def getBackgroundIntensity(self):
        sample = np.zeros(self.dim)
        sample = self.outline
        sample = sample/255 #outline was 255,now is 1
        sample = np.multiply(sample, self.raw)
        self.background_val = st.mode(sample[sample!=0].flatten())[0][0]
    def fillHoles(self):
        outline = self.outline
        outline = outline.astype(np.uint8)
        h, w = outline.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(outline, mask, (0,0), 255)
        outline = cv2.bitwise_not(outline)
        self.outline_filled = cv2.bitwise_xor(outline, self.outline)
        self.segmented = cv2.bitwise_xor(outline, self.outline)
    def erode(self):
        kernel = np.ones((self.erode_param,self.erode_param), np.uint8)
        self.outline_filled = cv2.erode(self.outline_filled, kernel) # moved from before block
    def skeletonize(self):
        outline = cv2.cvtColor(self.outline_filled,cv2.COLOR_GRAY2RGB)
        self.skeleton = cv2.ximgproc.thinning(cv2.cvtColor(outline, cv2.COLOR_RGB2GRAY))
        self.skeleton[-1] = 0
        self.skeleton[0] = 0
        self.skeleton[:,-1] = 0
        self.skeleton[:,0] = 0
    def sort_skeleton(self):
        points = np.column_stack(np.where(self.skeleton != 0)) #get all points of skeleton
        #self.points = points

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


        if xmax >= self.raw.shape[1] - 2: #touches right border 590

            start_point = points[xmaxarg,:]
            #print("right")
        if xmin <= 2: #touches left border, primary dims

            #print(xminarg)
            start_point = points[xminarg,:]
            #print("left")
            #print(start_point)
        if ymax >= np.shape(self.raw)[0] - 2:

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

    def spline_fit(self, body_points):
        x_vals = []
        y_vals = []
        t_vals = []

    ##Fit skeleton to spline curve
        for point_num in range(len(body_points)-1): #Arrange x, y values in increasing order
            x_vals.append(body_points[point_num][1]) #look here for flipped x,y
            y_vals.append(body_points[point_num][0])
            t_vals.append(point_num)
        t_vals.append(len(body_points)) # assign parameter t to each point
        x_vals.append(body_points[-1][1])
        y_vals.append(body_points[-1][0])
        spline_x = UnivariateSpline(t_vals, x_vals) # fit x values to spline
        global x_deriv, y_deriv
        x_deriv = spline_x.derivative() # x derivative
        spline_y = UnivariateSpline(t_vals,y_vals) #fit y values to spline
        y_deriv = spline_y.derivative() #y derivative

        #ax2.plot(spline_x(t_vals), spline_y(t_vals))
        self.spline_x = spline_x
        self.spline_y = spline_y
        self.spline_x_deriv = x_deriv
        self.spline_y_deriv = y_deriv
        self.t_vals = t_vals
        self.x_vals = x_vals
        self.y_vals = y_vals
    def vector(self, save):
        total_val = []
        t_max = max(self.t_vals)
        x1_vals = self.x_vals[:]
        y1_vals = self.y_vals[:]
        t1_vals = self.t_vals[:]
        x_vals = [] #x coordinate list of points to be sampled along arc
        y_vals = [] #y coordinate list of points to be sampled along arc
        t_vals = [] #t paramaterization
        ##Create sub-pixel sample points along skeleton
        for sample_pt in range(t_max*4): #currently we have one sample point per pixel gained from the skeleton, but since we have the spline fit we can sample subpixel points along arc.
            x_vals.append(float(self.spline_x(sample_pt*0.25)))
            y_vals.append(float(self.spline_y(sample_pt*0.25)))
            t_vals.append(sample_pt*0.25)
        #self.t_vals = t_vals
        #self.x_vals = x_vals
        #self.y_vals = y_vals
        #times = []
        #test = []
        arc_points = np.zeros(len(x_vals)) #arc_points is list of cumulative arc length
        arc_points[0] = 0
        #splx = UnivariateSpline(t_vals,x_vals)
        #sply = UnivariateSpline(t_vals,y_vals)
        #dx = splx.derivative()
        #dy = sply.derivative()
        radii = np.linspace(-25,25, 300)



        global flat_index, z, w
        flat_index = np.zeros((300,len(x_vals)))
        i = 0
        for point_num in range(len(x_vals)): #for every sample point along arc
            #print(len(x_vals))
            x = x_vals[point_num] # current x coordinate
            y = y_vals[point_num] # current y coordinate
            #if point_num == 1720:
                #print(x,y)
            if point_num ==0:
                arc_points[point_num] = 0
            else:
                arc_points[point_num] = arc_points[point_num-1] + math.sqrt((x-x_vals[point_num-1])**2 + (y-y_vals[point_num-1])**2) # populate cumulative arc length for each point
            t_current = t_vals[point_num]
            dy_dx = self.spline_y_deriv(t_current)/self.spline_x_deriv(t_current) #xy derivative gained from paramaterization... could result in error if denom is zero.
            #test.append(dy_dx)



            theta = np.arctan(dy_dx)
            theta_normal = theta + np.pi/2
            #want 150 sample points along normal line
            #15 px in both directions
            xs = radii * np.cos(theta_normal) + x
            ys = radii * np.sin(theta_normal) + y
            #plt.plot(xs,ys,"blue")
            #return
            floor_x = np.floor(xs-0.5)
            ceil_x = np.ceil(xs-0.5)
            floor_y = np.floor(ys-0.5)
            ceil_y = np.ceil(ys-0.5)

            #create block diagonal
            length = len(xs) * 2
            diag_mat = np.zeros((length, length))

            #global evens, lin
            evens = np.linspace(0,length-2, int(length/2))
            odds = evens + 1

            evens = evens.astype(int)
            odds = odds.astype(int)

            floor_x = floor_x.astype(int)
            floor_y = floor_y.astype(int)
            ceil_x = ceil_x.astype(int)
            ceil_y = ceil_y.astype(int)
            #for i in range(len(xs)):
                #print(i)
                #diag_mat[i*2, i*2] = img[int(floor_y[i]),int(floor_x[i])]
                #diag_mat[i*2, i*2+1] = img[int(ceil_y[i]),int(floor_x[i])]
                #diag_mat[i*2+1, i*2] = img[int(floor_y[i]),int(ceil_x[i])]
                #diag_mat[i*2+1, i*2+1] = img[int(ceil_y[i]),int(ceil_x[i])]
            diag_mat[evens, evens] = self.raw[floor_y,floor_x]
            diag_mat[evens, odds] = self.raw[ceil_y,floor_x]
            diag_mat[odds, evens] = self.raw[floor_y,ceil_x]
            diag_mat[odds, odds] = self.raw[ceil_y,ceil_x]
            #create flattened array
            xs = xs-0.5-floor_x
            ys = ys -0.5-floor_y
            one_minus_xs = 1-xs
            one_minus_ys = 1-ys
            interp_xs = np.array((one_minus_xs, xs)).T.flatten()#is two stacked on top of eo
            interp_ys = np.array((one_minus_ys, ys)).T.flatten()
            total_val.append(np.dot(interp_xs @ diag_mat, interp_ys))
            ##For visualizing spread of intensity
            if save == True:
                diag_y_mat = np.zeros((length, int(length/2)))
                lin = np.linspace(0,int(length/2)-1, int(length/2))
                lin = lin.astype(int)
                diag_y_mat[evens,lin] = interp_ys[evens]
                diag_y_mat[odds,lin] = interp_ys[odds]
                smoothed_signal = signal.savgol_filter(interp_xs @ diag_mat @ diag_y_mat, 50, 3)
                #plt.plot(smoothed_signal)
                #plt.ylim(0,40)
                #plt.savefig(str(point_num)+".png")
                #plt.close()
                #regress = st.linregress(np.linspace(1,300, 300), smoothed_signal)
                #flat_index[i] = regress.rvalue**2
                #if point_num == 1390:
                #    print(x,y)
                #    plt.plot(smoothed_signal)
                #    plt.plot(np.linspace(0,300,299), regress.slope*np.linspace(0,300,299) + np.ones(299)*regress.intercept)
                #    plt.show()
                flat_index[:,i] = interp_xs @ diag_mat @ diag_y_mat
                #flat_index[:,i] = smoothed_signal
                #diff = np.abs(np.diff(smoothed_signal))
                #flat_index[i] = np.sum(diff)
                #plt.close()
                #times.append(time.time())
            i+=1
        total_val = np.asarray(total_val)
        total_val = total_val - self.background_val * 2* int(15/0.1) #normalization factor
       # return [total_val, arc_points, arc_points.max(), x_meshgrid, y_meshgrid, intensity_meshgrid]
        mode = st.mode(flat_index.flatten())[0][0] + 4
        flat_index[flat_index < mode] = 0
        w = np.count_nonzero(flat_index, axis = 0)
        intestine_ind = np.argmax(w) #this is the index of intestine


        smoothed_w = signal.savgol_filter(w, 10, 3)
        #intestine_ind = np.argmax(np.diff(smoothed_w))
        intestine_ind = np.argmax(smoothed_w[len(smoothed_w)//2:]) + (len(smoothed_w) // 2)
        # print(intestine_ind)

        #plt.plot(intestine_ind, smoothed_w[intestine_ind] , "x")
        #plt.plot(arc_points,smoothed_w, label = str(j))

        #Interpolate wrt arc length first to get common x axis
        #must run sequentially now
        interpolated_widthform = interp1d(arc_points,smoothed_w)
        ordered_x = np.linspace(1, math.floor(arc_points.max()), math.floor(arc_points.max()))
        ordered_width = interpolated_widthform(ordered_x)

        prev_width = self.prev_wave.copy()

        #Now correlate the two
        lag = None
        if j != '00000':
            print(intestine_ind, arc_points[intestine_ind])

            #Set everything else but intestine region to zero of current widthform
            ordered_width_intestine_region = ordered_width.copy()
            ordered_width_intestine_region[:math.floor(arc_points[intestine_ind]) - 15] = 0
            ordered_width_intestine_region[math.floor(arc_points[intestine_ind]) + 90:] = 0
            # Set the first half of the previous widthform to zero to prevent head of worm from distracting
            prev_width[:len(prev_width)//2] = 0

            #plt.plot(ordered_width_intestine_region, label = "intestine region")
            #plt.plot(prev_width, label = "prev width")
            #plt.plot(ordered_width, label = "cur width")

            corr = signal.correlate((prev_width), (ordered_width_intestine_region),mode="full")

            lags = signal.correlation_lags(prev_width.size, ordered_width_intestine_region.size, mode='full')
            lag = lags[np.argmax(corr)]

            #Shift the arc length indices and see if there is drift
            # if lag > 0:
            #     ordered_width = ordered_width[:-lag] #delete lag number of entries at end
            #     ordered_width = np.concatenate((np.zeros(lag),ordered_width),axis=0) # if lagging, need separate case if leading
            # if lag < 0:
            #     #plt.plot(self.waveform[-lag:])
            #     #plt.show()
            #     ordered_width = ordered_width[-lag:]
            #     ordered_width = np.concatenate((ordered_width,np.zeros(-lag)),axis=0)
            # plt.plot(prev_width, label = "prev")
            # plt.plot(ordered_width, label = "curr")
            # plt.legend()
            # plt.show()
            print(lag) # how much arc length should be shifted



            #intestine_ind = arc_points[intestine_ind]
        return [total_val, arc_points, arc_points.max(), lag, ordered_width]

    def InterpolateIntensity(self, total_val, xpoints, max_arc_length,spacing): #only for first frame now
        if xpoints[0] > 0:
            initial_l = xpoints[0]
            xpoints = np.pad(xpoints, [(math.ceil(initial_l),0)],mode = 'linear_ramp')
            total_val = np.pad(total_val, [(math.ceil(initial_l),0)])

        diff = math.ceil(max_arc_length - xpoints.max()) #max arc length for first frame is always gonna be lower by 10%. Diff is positive
        if diff > 0: #fill with zeros
            xpoints = np.pad(xpoints, [(0,diff)], mode='linear_ramp',end_values = ((0,xpoints.max() + diff)))
            total_val = np.pad(total_val, [(0,int(diff))])
            #print("change?")
        #plt.plot(xpoints,total_val)
        #plt.show()
        func = interp1d(xpoints, total_val)
        '''Change based on expected length'''
        xnew = np.linspace(1,max_arc_length,max_arc_length*spacing)

        row_vec[200:max_arc_length*spacing+200] = func(xnew)
        frame.waveform = row_vec.copy()


    def AlignWaveform(self, first_intes_ind, intes_ind): #Do both at the same time
#Length correction part (if current frame length is shorter than reference, FILL WITH ZEROS!
#Aligning part
        ref = first_intes_ind
        lag = ref - intes_ind
        print(lag)
        if lag > 0:
            self.waveform = self.waveform[:-lag] #delete lag number of entries at end
            self.waveform = np.concatenate((np.zeros(lag),self.waveform),axis=0) # if lagging, need separate case if leading
        if lag < 0:
            #plt.plot(self.waveform[-lag:])
            #plt.show()
            self.waveform = self.waveform[-lag:]
            self.waveform = np.concatenate((self.waveform, np.zeros(-lag)),axis=0)







#time_series = np.zeros([1000,400]) #Empty kymograph: 349 body points/1000 frames

#os.chdir("Kymographs")
#os.chdir("..")
#os.chdir("Kymographs")
arr = np.zeros(np.shape(f['t00000']['s00']['0']['cells'])[1:])
j = '00000'
for i in range(np.shape(f['t00000']['s00']['0']['cells'])[0]): #Sum up z stack
  t = f['t'+ j]['s00']['0']['cells']
  arr = arr + t[i] # arr holds the summed up z stack image
arr = arr.T #for test only#start_time = time.time()
original_dim = arr.shape
original_arr = np.array(arr)


#print(arr*30)

    #plt.imshow(arr.squeeze())


  #cv2_imshow(segnet(img_tensor).detach().cpu().numpy().squeeze()*255)'


##For first frame
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
frame.outline = functions.ChooseLargestBlob(frame.outline)
#frame.outline = frame.outline.astype(np.uint8)
frame.fillHoles()
frame.erode()
frame.skeletonize()

body_points = frame.sort_skeleton()

frame.raw = np.concatenate((frame.raw, np.zeros((30, frame.raw.shape[1]))),axis=0)
frame.raw = np.concatenate((frame.raw, np.zeros((frame.raw.shape[0],30))),axis=1)


frame.spline_fit(body_points)
frame.prev_wave = []

global first_intestine_ind

[total_val, xpoints, cur_arc_length, lag, widthform] = frame.vector(save = True)

total_valf = total_val.copy()
xpointsf = xpoints.copy()

global max_arc_length
max_arc_length = math.floor(0.9*xpoints.max()) #first frame sets the kymograph y dim for all frames.
global time_series
time_series = np.zeros([total_frames, max_arc_length*spacing+400])
global row_vec
row_vec = np.zeros((max_arc_length*spacing+400))



frame.InterpolateIntensity(total_val, xpoints, max_arc_length, spacing)
kymo_length = max_arc_length *spacing

time_series[0] = frame.waveform
frame.kymo_length = kymo_length

for j in range(30,31): #Loop over how many frames you want to analyse
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

    frame.prev_wave = widthform

    [total_val, xpoints, cur_arc_length, lag, widthform] = frame.vector(save=True)


    xpoints = xpoints + lag
    frame.InterpolateIntensity(total_val, xpoints, max_arc_length, spacing)

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

