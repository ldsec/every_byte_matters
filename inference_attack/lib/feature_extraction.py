import csv
import glob
import math
import os
import pickle
import sys
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels


# --------------------------------------
# Extract Features

def extract_features(xy, capture_duration_does_nothing=0): # dataset is {'xs': [packet1, packet2,...], 'ys': [packet1, packet2,...]} where x is time and y is size
    xs = xy['xs']
    ys = xy['ys']
    f = dict()

    def take(arr, n=30):
        if len(arr) > n:
            return arr[:30]
        return arr

    def stats(key, data):
        if len(data) == 0:
            data=[-1]
        f['min_'+key] = np.min(data)
        f['mean_'+key] = np.mean(data)
        f['max_'+key] = np.max(data)
        f['count_'+key] = len(data)
        f['std_'+key] = np.std(data)
        f['kurtosis_'+key] = kurtosis(data)


    # general statistics
    stats("non_null", [abs(y) for y in ys if y != 0])
    stats("outgoing", [abs(y) for y in ys if y > 0])
    stats("incoming", [abs(y) for y in ys if y < 0])
    stats("outgoing_30", [abs(y) for y in take(ys) if y > 0])
    stats("incoming_30", [abs(y) for y in take(ys) if y < 0])

    # statistics about timings
    x_deltas = []
    i = 1
    while i<len(xs):
        x_deltas.append(xs[i]-xs[i-1])
        i += 1

    stats("x_deltas", x_deltas)
    stats("x_deltas_30", take(x_deltas))
    
    # bursts

    # unique packet lengths [Liberatore and Levine; Herrmann et al.]
    lengths = dict()
    for i in range(500,1000):
        lengths[str(i)] = 0
    for y in ys:
        if str(abs(y)) in lengths:
            lengths[str(abs(y))] += 1

    lengths_array = list(lengths.values())
    stats("unique_lengths", lengths_array)
    for l in lengths:
        f['unique_lengths_'+str(l)] = lengths[l]


    return f

    # TaoWang14:
    # packet ordering (for each outgoing packet, add number indicating the total number of incoming packet between this packet and the previous outgoing one)
    # concentration of outgoing packets (count number of outgoing packets in non-overlapping spans of 30 packets)
    # bursts (sequence of outgoing with no incoming); add max, mean burst length; add number of bursts as feature
    # initial lengths (initial N=20 packet sizes)
    # note: for variable-length features (e.g., packet ordering), then pad to predefined number with special character 'X'

    # CUMUL:
    # Trace of packet T=(p1,p2...pN)
    # C(T) = ((0,0), (a1,c1), (aN,cN))
    # where c1=p1, a1=|p1|, ci=c(i-1)+pi, ai=a(i-1) + |pi|
    # then, linear interpolation of C(T), and pick n additional features at equidistance

    # k-Fingerprinting:
    #1.   Number of incoming packets.
    #2.   Number of outgoing packets as a fraction of the total numberof packets.
    #3.   Number of incoming packets as a fraction of the total numberof packets.
    #4.   Standard deviation of the outgoing packet ordering list.
    #5.   Number of outgoing packets.
    #6.   Sum of all items in the alternative concentration feature list.
    #7.   Average of the outgoing packet ordering list.
    #8.   Sum of incoming, outgoing and total number of packets.
    #9.   Sum of alternative number packets per second.
    #10.   Total number of packets.
    #11-18.   Packet concentration and ordering features list.
    #19.   The total number of incoming packets stats in first 30 packets.
    #20.   The total number of outgoing packets stats in first 30 packets.

def extract_features_short(xy, adversary_capture_duration = -1): # dataset is {'xs': [packet1, packet2,...], 'ys': [packet1, packet2,...]} where x is time and y is size
    
    #xs = xy['xs']
    #ys = xy['ys']
    
    xs = []
    ys = []
    i = 0
    while i < len(xy['ys']) and (adversary_capture_duration == -1 or xy['xs'][i]<adversary_capture_duration):
        xs.append(xy['xs'][i])
        ys.append(xy['ys'][i])
        i += 1
    
    f = dict()

    def take(arr, n=30):
        if len(arr) > n:
            return arr[:30]
        return arr

    def stats(key, data):
        if len(data) == 0:
            data=[-1]
        f['min_'+key] = np.min(data)
        f['mean_'+key] = np.mean(data)
        f['max_'+key] = np.max(data)
        f['count_'+key] = len(data)
        f['std_'+key] = np.std(data)
        f['kurtosis_'+key] = kurtosis(data)


    # general statistics
    stats("non_null", [abs(y) for y in ys if y != 0])
    stats("outgoing", [abs(y) for y in ys if y > 0])
    stats("incoming", [abs(y) for y in ys if y < 0])

    # statistics about timings
    x_deltas = []
    i = 1
    while i<len(xs):
        x_deltas.append(xs[i]-xs[i-1])
        i += 1

    stats("x_deltas", x_deltas)
    
    return f

def extract_features_new(xy, adversary_capture_duration=-1): # dataset is {'xs': [packet1, packet2,...], 'ys': [packet1, packet2,...]} where x is time and y is size
    
    xs = []
    ys = []
    i = 0
    while i < len(xy['ys']) and (adversary_capture_duration == -1 or xy['xs'][i]<adversary_capture_duration):
        xs.append(xy['xs'][i])
        ys.append(xy['ys'][i])
        i += 1

    f = dict()

    def deltas(serie):
        out = []
        i = 1
        while i<len(serie):
            out.append(serie[i]-serie[i-1])
            i += 1
        return out
    
    def take(arr, n=30):
        if len(arr) > n:
            return arr[:30]
        return arr

    def sum(arr):
        x = 0
        for v in arr:
            x += v
        return x

    def stats(key, data):
        if len(data) == 0:
            data=[-1]
        f['min_'+key] = np.min(data)
        f['mean_'+key] = np.mean(data)
        f['max_'+key] = np.max(data)
        f['count_'+key] = len(data)
        f['std_'+key] = np.std(data)
        f['kurtosis_'+key] = kurtosis(data)

    def avgIPT(xs, ys, incoming=True):
        def condition(y):
            if incoming and y>=0:
                return True
            if not incoming and y<0:
                return True
            return False
        zipped = [xy for xy in zip(xs, ys) if condition(xy[1])]
        P = len(zipped)

        if P == 1:
            return zipped[0][1]
        nom = sum(deltas([xy[0] for xy in zipped]))

        return nom/(P-1)

    f['av_delta_inc'] = avgIPT(xs, ys, True)
    f['av_delta_out'] = avgIPT(xs, ys, False)

    count_send = len([abs(y) for y in ys if y > 0])
    count_receive = len([abs(y) for y in ys if y < 0])
    
    if count_send + count_receive == 0:     
        f['PCR_send'] = 0
        f['PCR_receive'] = 0
    else:
        f['PCR_send'] = count_send / (count_send + count_receive)
        f['PCR_receive'] = count_receive / (count_send + count_receive)

    sum_send = sum([abs(y) for y in ys if y > 0])
    sum_receive = sum([abs(y) for y in ys if y < 0])


    if sum_send + sum_receive == 0:     
        f['DSR_send'] = 0
        f['DSR_receive'] = 0
    else:
        f['DSR_send'] = sum_send / (sum_send + sum_receive)
        f['DSR_receive'] = sum_receive / (sum_send + sum_receive)

    # statistics about timings
    x_deltas = []
    i = 1
    while i<len(xs):
        x_deltas.append(xs[i]-xs[i-1])
        i += 1

    stats("non_null", [abs(y) for y in ys if y != 0])
    stats("outgoing", [abs(y) for y in ys if y > 0])
    stats("incoming", [abs(y) for y in ys if y < 0])
    stats("outgoing_30", [abs(y) for y in take(ys) if y > 0])
    stats("incoming_30", [abs(y) for y in take(ys) if y < 0])
    stats("x_deltas", x_deltas)
    
    # bursts


    # unique packet lengths [Liberatore and Levine; Herrmann et al.]
    lengths = dict()
    for i in range(1,10):
        lengths[str(i)] = 0

    for y in ys:
        i = 0
        y = abs(y)
        i = y // 10
        if i > 10:
            i == 10
        if str(i) in lengths:
            lengths[str(i)] += 1

    for k in lengths:
        f[k] = lengths[k]

    return f

def extract_features_dual(xy, adversary_capture_duration = -1): # dataset is {'xs': [packet1, packet2,...], 'ys': [packet1, packet2,...]} where x is time and y is size
    
    xs = xy['xs']
    ys = xy['ys']

    mid = len(xs) // 2
    xs1 = xs[:mid]
    ys1 = ys[:mid]
    xs2 = xs[mid:]
    ys2 = ys[mid:]

    f = extract_features_new(xy)
    if len(xs1) > 0:
        f1 = extract_features_new(dict(xs=xs1, ys=ys1))
        for k in f1:
            f['start_'+k] = f1[k]

    if len(xs2) > 0:
        f2 = extract_features_new(dict(xs=xs2, ys=ys2))
        for k in f2:
            f['end_'+k] = f2[k]
    return f


def extract_features_dual_adv(xy, adversary_capture_duration = -1): # dataset is {'xs': [packet1, packet2,...], 'ys': [packet1, packet2,...]} where x is time and y is size
    
    xs = xy['xs']
    ys = xy['ys']

    if adversary_capture_duration > -1 :
        # only keep the first X seconds
        xs = []
        ys = []
        i = 0
        while i < len(xy['ys']) and xy['xs'][i]<adversary_capture_duration:
            xs.append(xy['xs'][i])
            ys.append(xy['ys'][i])
            i += 1

        # if X less than 10s, no point in cutting in two
        if adversary_capture_duration < 10:
            return extract_features_new(dict(xs=xs, ys=ys))

    
    # cut in two parts, extract stats on: global, part1, part2
    mid = len(xs) // 2
    xs1 = xs[:mid]
    ys1 = ys[:mid]
    xs2 = xs[mid:]
    ys2 = ys[mid:]
    
    f = extract_features_new(xy)
    if len(xs1) > 0:
        f1 = extract_features_new(dict(xs=xs1, ys=ys1))
        for k in f1:
            f['start_'+k] = f1[k]

    if len(xs2) > 0:
        f2 = extract_features_new(dict(xs=xs2, ys=ys2))
        for k in f2:
            f['end_'+k] = f2[k]
    return f