#Imported packages for graphing
from __future__ import division
import h5py
import numpy as np
from zmq_client import adc_to_voltage
import sys
from scipy.optimize import fmin

print "start program"

#Finds the amplitude of each of the curves with voltage barrier
def find_amp(v):
        amplitude = np.min(v,axis=1)
        filteramp = amplitude[amplitude < -200]
        return abs(amplitude)

#Finds the time resolution of the associated voltages
def find_res(v):
    t = np.empty(v.shape[0],dtype=float)
    for i in range(len(v)):
        if i % 100 == 0:
            print "\r%i/%i" % (i+1,len(v)),
            sys.stdout.flush()
        j = np.argmin(v[i])
        disc = v[i,j]*0.4
        while v[i,j] < disc and j > 0:
            j -= 1
        t[i] = j + (disc - v[i,j])/(v[i,j+1] - v[i,j])
    print()
    return t

#Takes the input from the terminal and reads it
if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help="input filename")
    args = parser.parse_args()

    f = h5py.File(args.filename)
    dset = f['c1'][:100000]
    amp = find_amp(dset)
    f_dset = dset[amp > 200]

    t1 = find_res(f_dset)
    t = t1.copy()
    t *= 0.5
    t -= np.mean(t)

    dset2 = f['c2'][:100000]
    amp2 = find_amp(dset2)
    f_dset2 = dset[amp > 200]

    t0 = find_res(f_dset2)
    t2 = t0.copy()
    t2 *= 0.5
    t2 -= np.mean(t)

    plt.scatter(f_dset, t, color='blue')
    plt.scatter(f_dset2, t2, color='red')
    plt.xlabel("Amplitude")
    plt.ylabel("Time Resolution")
    plt.title("Time Resolution vs. Amplitude")

plt.show() 
