#!/usr/bin/env python
import collections, itertools
import numpy as np
from scipy.signal import lfilter
import filterpy.kalman as kf

class KalmanFilter(object):
    def __init__(self,
                 F=None,
                 H=None,
                 P=None,
                 R=None,
                 Q=None,
                 init_val=0):
        '''Initialize'''
        self.f = kf.KalmanFilter(dim_x=1, dim_z=1)
        self.f.x = np.array([init_val])
        self.f.F = np.array([[F]]) if F is not None else np.array([[1.]])
        self.f.H = np.array([[H]]) if H is not None else np.array([[1.]])
        self.f.P = np.array([[P]]) if P is not None else np.array([[10e-4]])
        self.f.R = np.array([[R]]) if R is not None else np.array([[10e-5]])
        self.f.Q = np.array([[Q]]) if Q is not None else np.array([[10e-6]])

    def reinit(self, init_val):
        '''Reset filter init values'''
        self.f.x = np.array([init_val])

    def filt(self, new_datum):
        
        self.f.z = new_datum
        self.f.x, self.f.P = kf.predict(self.f.x, self.f.P, self.f.F, self.f.Q)
        self.f.x, self.f.P = kf.update(self.f.x, self.f.P, self.f.z, self.f.R, self.f.H)

        return self.f.x[0]

class OnlineButter(object):
    '''Lightweight implementation for online filtering with Butterworth filter'''

    def __init__(self,
                 b=None,
                 a=None,
                 init_val=0):
        '''Initialize'''
        self.b = b if b is not None else [0.02008337, 0.04016673, 0.02008337]
        self.a = a if a is not None else [1., -1.56101808, 0.64135154]

        self.raw = collections.deque([init_val]*3, maxlen=3)
        self.filtered = collections.deque([init_val]*3, maxlen=3)

    def reinit(self, init_val):
        '''Reset filter init values'''
        self.raw = collections.deque([init_val]*3, maxlen=3)
        self.filtered = collections.deque([init_val]*3, maxlen=3)

    def filt(self, new_datum):
        '''Applies a 2nd order butteworth filter'''

        self.raw.append(new_datum)

        self.filtered.append(self.b[0] * self.raw[2] \
                         + self.b[1] * self.raw[1] \
                         + self.b[2] * self.raw[0] \
                         - self.a[1] * self.filtered[2] \
                         - self.a[2] * self.filtered[1])

        return self.filtered[2]

class OnlineButterSixthOrder(object):
    '''Lightweight implementation for online filtering with Butterworth filter'''

    def __init__(self,
                 b=None,
                 a=None,
                 init_val=0):
        '''Initialize'''
        self.b = np.array(b) if b is not None else np.array([0.00000000085315951525721800408064154908061,
                                          0.0000000051189570915433080244838492944837,
                                          0.000000012797392728858270061209623236209,
                                          0.000000017063190305144360081612830981612,
                                          0.000000012797392728858270061209623236209,
                                          0.0000000051189570915433080244838492944837,
                                          0.00000000085315951525721800408064154908061])
        self.a = np.array(a) if a is not None else np.array([1.0,
                                          - 5.757244186246575523568935750518,
                                          13.815510806058025394804644747637,
                                          -17.68737617989402721718761313241,
                                          12.741617329229226740494596015196,
                                          - 4.896924891433742210722357413033,
                                          0.78441717688930268082003749441355])
        

        self.raw = collections.deque([init_val]*len(self.a), maxlen=len(self.a))
        self.filtered = collections.deque([init_val]*len(self.a), maxlen=len(self.a))

    def reinit(self, init_val):
        '''Reset filter init values'''
        self.raw = collections.deque([init_val]*len(self.a), maxlen=len(self.a))
        self.filtered = collections.deque([init_val]*len(self.a), maxlen=len(self.a))

    def filt(self, new_datum):
        '''Applies a 2nd order butteworth filter'''

        self.raw.append(new_datum)
        
        self.filtered = lfilter(self.b, self.a, self.raw)

        # filtered = self.b[0]*self.raw[0]

        # tmp = self.b[1:]*list(itertools.islice(self.raw, 1, len(self.raw))) - self.a[1:]*list(itertools.islice(self.filtered, 0, len(self.filtered)-1))
        # tmp = self.b*self.raw - self.a*self.filtered
        # filtered = sum(tmp)

        # for k in range(1, len(self.a)):
        #     tmp += self.b[k]*self.raw[len(self.a)-k-1] - self.a[k]*self.filtered[len(self.a)-k]
        
        # self.filtered.appendleft(filtered)

        return self.filtered[-1]

class MovingAverage:
    def __init__(self, window, init_val=0):
        self.window = window
        self.raw = collections.deque([init_val]*self.window, maxlen=self.window)
        self.filtered = collections.deque([init_val]*self.window, maxlen=self.window)

    def filt(self, new_datum):
        '''Applies a moving average filter'''

        self.raw.append(new_datum)
        
        self.filtered = np.cumsum(self.raw)/self.window
    
        return self.filtered[-1]