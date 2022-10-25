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

def fitSpline(skeleton_points):
    x_vals = []
    y_vals = []
    t_vals = []

##Fit skeleton to spline curve
    for point_num in range(len(skeleton_points)-1): #Arrange x, y values in increasing order
        x_vals.append(skeleton_points[point_num][1]) #look here for flipped x,y
        y_vals.append(skeleton_points[point_num][0])
        t_vals.append(point_num)
    t_vals.append(len(skeleton_points)) # assign parameter t to each point
    x_vals.append(skeleton_points[-1][1])
    y_vals.append(skeleton_points[-1][0])
    spline_x = UnivariateSpline(t_vals, x_vals) # fit x values to spline
    #global x_deriv, y_deriv
    x_deriv = spline_x.derivative() # x derivative
    spline_y = UnivariateSpline(t_vals,y_vals) #fit y values to spline
    y_deriv = spline_y.derivative() #y derivative

    #ax2.plot(spline_x(t_vals), spline_y(t_vals))

    return [spline_x, spline_y, t_vals, x_vals, y_vals]

def integrateImage(image, spline_x, spline_y, t_vals, x_vals, y_vals):

        x_deriv = spline_x.derivative()
        y_deriv = spline_y.derivative()
        total_val = []
        average_val = []
        t_max = max(t_vals)
        x1_vals = x_vals[:]
        y1_vals = y_vals[:]
        t1_vals = t_vals[:]
        x_vals = [] #x coordinate list of points to be sampled along arc
        y_vals = [] #y coordinate list of points to be sampled along arc
        t_vals = [] #t paramaterization
        ##Create sub-pixel sample points along skeleton
        for sample_pt in range(t_max*4): #currently we have one sample point per pixel gained from the skeleton, but since we have the spline fit we can sample subpixel points along arc.
            x_vals.append(float(spline_x(sample_pt*0.25)))
            y_vals.append(float(spline_y(sample_pt*0.25)))
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
            dy_dx = y_deriv(t_current)/x_deriv(t_current) #xy derivative gained from paramaterization... could result in error if denom is zero.
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
            diag_mat[evens, evens] = image[floor_y,floor_x]
            diag_mat[evens, odds] = image[ceil_y,floor_x]
            diag_mat[odds, evens] = image[floor_y,ceil_x]
            diag_mat[odds, odds] = image[ceil_y,ceil_x]
            #create flattened array
            xs = xs-0.5-floor_x
            ys = ys -0.5-floor_y
            one_minus_xs = 1-xs
            one_minus_ys = 1-ys
            interp_xs = np.array((one_minus_xs, xs)).T.flatten()#is two stacked on top of eo
            interp_ys = np.array((one_minus_ys, ys)).T.flatten()
            total_val.append(np.dot(interp_xs @ diag_mat, interp_ys))

            ##For visualizing spread of intensity
            diag_y_mat = np.zeros((length, int(length/2)))
            lin = np.linspace(0,int(length/2)-1, int(length/2))
            lin = lin.astype(int)
            diag_y_mat[evens,lin] = interp_ys[evens]
            diag_y_mat[odds,lin] = interp_ys[odds]
            smoothed_signal = signal.savgol_filter(interp_xs @ diag_mat @ diag_y_mat, 50, 3)

            flat_index[:,i] = interp_xs @ diag_mat @ diag_y_mat

            i+=1
        total_val = np.asarray(total_val)
        #total_val = total_val - background_val * 2* int(15/0.1) #normalization factor
       # return [total_val, arc_points, arc_points.max(), x_meshgrid, y_meshgrid, intensity_meshgrid]
        mode = st.mode(flat_index.flatten())[0][0] + 4
        flat_index[flat_index < mode] = 0
        w = np.count_nonzero(flat_index, axis = 0)

        smoothed_w = signal.savgol_filter(w, 10, 3)
        #intestine_ind = np.argmax(np.diff(smoothed_w))
        intestine_ind = np.argmax(smoothed_w[len(smoothed_w)//3:]) + (len(smoothed_w) // 3)

        print(intestine_ind)
        return [total_val, arc_points, arc_points.max(), intestine_ind]




#intestine_start_pos = 470
#intestine_max_pos = 570


def correlation_lags(in1_len, in2_len, mode='full'):
    # calculate lag ranges in different modes of operation
    if mode == "full":
        # the output is the full discrete linear convolution
        # of the inputs. (Default)
        lags = np.arange(-in2_len + 1, in1_len)
    elif mode == "same":
        # the output is the same size as `in1`, centered
        # with respect to the 'full' output.
        # calculate the full output
        lags = np.arange(-in2_len + 1, in1_len)
        # determine the midpoint in the full output
        mid = lags.size // 2
        # determine lag_bound to be used with respect
        # to the midpoint
        lag_bound = in1_len // 2
        # calculate lag ranges for even and odd scenarios
        if in1_len % 2 == 0:
            lags = lags[(mid-lag_bound):(mid+lag_bound)]
        else:
            lags = lags[(mid-lag_bound):(mid+lag_bound)+1]
    elif mode == "valid":
        # the output consists only of those elements that do not
        # rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
        # must be at least as large as the other in every dimension.

        # the lag_bound will be either negative or positive
        # this let's us infer how to present the lag range
        lag_bound = in1_len - in2_len
        if lag_bound >= 0:
            lags = np.arange(lag_bound + 1)
        else:
            lags = np.arange(lag_bound, 1)
    return lags



def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)



def InterpolateIntensity(total_val, xpoints, max_arc_length,spacing): #only for first frame now
    row_vec = np.zeros((max_arc_length*spacing+400))
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
    return row_vec.copy()
