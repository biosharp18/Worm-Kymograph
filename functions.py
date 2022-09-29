import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import cv2


def ChooseLargestBlob(image):
   # [nb_components, output, stats, centroids, sizes] = frame.identifyComponents(frame.thresh_img)
    connectivity = 4  #4-connectivity
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    t = np.zeros(image.shape)
    t[output == np.argmax(sizes)+1] = 255
    t = t.astype(np.uint8)
    return t


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


def RealignKymo(kym, intestine_start_ind, intestine_max_ind):
    #global newkym
    #first do cross correlation with entire waveform, then search for intestine area
    newkym = np.zeros(kym.shape)
    ref = kym[:,0].copy()
    for colnum in range(newkym.shape[1]):
        #print(colnum)

        val1 = ref
        val2 = kym[:,colnum].copy()
        corr = signal.correlate(val1, val2,mode="full")

        lags = signal.correlation_lags(val1.size, val2.size, mode='full')
        ##Mean squared error
        #lags,corr = MeanSquaredError(val1,val2, mode= 'full')
        lag = lags[np.argmax(corr)]
        if lag == 0:
            newcol = kym[:,colnum]
        if lag > 0:
            val2 = val2[:-lag] #delete lag number of entries at end
            val2 = np.concatenate((np.zeros(lag),val2),axis=0)
            newcol = kym[:-lag,colnum] # if lagging, need separate case if leading
            newcol = np.concatenate((np.zeros(lag),newcol),axis=0)
        if lag < 0:
            #plt.plot(self.waveform[-lag:])
            #plt.show()
            val2 = val2[-lag:]
            val2 = np.concatenate((val2, np.zeros(-lag)),axis=0)
            newcol = kym[-lag:,colnum]
            newcol = np.concatenate((newcol, np.zeros(-lag)),axis=0)
        newkym[:,colnum] = newcol




        val1[:intestine_start_ind] = val1[intestine_start_ind]#change max intestine pos
        #val1 = np.diff(val1)
        #val1 = self.prev_wave[intestine_start_ind:]

        #first do realignment of entire waveform



        min_intestine = val2[intestine_start_ind:intestine_max_ind].min()
        #print(np.where(val2 == min_intestine))
        global m, v
        m = min_intestine
        v = val2
        min_intestine_start_ind = int(np.where(val2 == min_intestine)[-1]) #last


        val2[:min_intestine_start_ind] = min_intestine #the lower half of intestine
        #val2 = np.diff(val2)

        #val2 = self.waveform[intestine_start_ind:]
        ##Cross correlation
        corr = signal.correlate(val1,val2, mode='full')
        lags = signal.correlation_lags(val1.size, val2.size, mode='full')
        ##Mean squared error
        #lags,corr = MeanSquaredError(val1,val2, mode= 'full')
        lag = lags[np.argmax(corr)] #second lags first by p
        if lag == 0:
            newcol = newkym[:,colnum].copy()
        if lag > 0:
            newcol = newkym[:-lag,colnum] #delete lag number of entries at end
            newcol = np.concatenate((np.zeros(lag),newcol),axis=0) # if lagging, need separate case if leading
        if lag < 0:
            #plt.plot(self.waveform[-lag:])
            #plt.show()
            newcol = newkym[-lag:,colnum]
            newcol = np.concatenate((newcol, np.zeros(-lag)),axis=0)
        newkym[:,colnum] = newcol
    return newkym


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
