"""MACHI - Minimal Atmospheric Compensation for Hyperspectral Imagers.

Vendored, unchanged in behaviour, from https://github.com/NTNU-SmallSat-Lab/MACHI
(machi/base.py at commit ebbc3e72, 2026-01-29), by Joseph Garrett (NTNU SmallSat
Lab), under the MIT licence below. Only the functions the correction needs are
included (atm_correction and its helpers); upstream's AC class, plotting and
solar-spectrum loader are not.

Model: R_surface = (R_TOA - A) / (solar T), with one scattering term A and one
transmission T per band for the whole image. A starts from the per-band minimum
(dark pixel subtraction), T from 1 - 2 min; T and A are then updated alternately
on the odd and even bands to minimise the image-averaged squared spectral
derivative of the corrected spectra (surface spectra are smooth, gas absorption
is not). See upstream's README for its known limitations.

hypso.processing.atmospheric's "machi" method wraps this for products.

MIT License

Copyright (c) 2024 NTNU SmallSat Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np


def min_smoother(mins):
    mins_out = np.zeros_like(mins, dtype=np.float32)
    mins_out[0] = mins[0]
    for i in range(len(mins)-1):
        mins_out[i+1] = np.minimum(mins[i+1], mins_out[i])
    return mins_out


def T_update(raw_data, lattice, A, T, weights=1, solar=1):
    '''
    update t
    '''
    active_bands = list(lattice.keys())
    st = solar*T
    top_2sum = np.sum(weights*((raw_data[:,active_bands]-A[active_bands])**2).T, axis=1)
    passive_bands1 = [lattice[b][0] for b in active_bands]
    bottom_sum1 = np.sum(weights*((raw_data[:,active_bands]-A[active_bands])\
                         *(raw_data[:,passive_bands1]-A[passive_bands1])/st[passive_bands1]).T,
                         axis=1)

    active_bands_2 = [a for a in active_bands if len(lattice[a])>1]
    passive_bands_2 = [lattice[b][1] for b in active_bands_2]

    bottom_sum2 = np.sum(weights*((raw_data[:,active_bands_2]-A[active_bands_2])\
                         *(raw_data[:,passive_bands_2]-A[passive_bands_2])/st[passive_bands_2]).T, axis=1)

    num = np.zeros(len(lattice))
    denom = np.zeros(len(lattice))
    two_neighbors = [i in active_bands_2 for i in active_bands]

    num[:] = top_2sum
    num[two_neighbors] += top_2sum[two_neighbors]
    denom[:] = bottom_sum1[:]
    denom[two_neighbors] += bottom_sum2[:]
    new_T = 1/solar[active_bands]*num/denom
    new_T[new_T > 1] = 1
    return new_T


def A_update(raw_data, lattice, A, T, band_mins, weights=1, solar=1):
    active_bands = list(lattice.keys())
    st = solar*T
    
    first_sum = np.sum(weights*raw_data[:,active_bands].T, axis=1)
    
    passive_bands1 = [lattice[b][0] for b in active_bands]
    second_sum1 = np.sum(weights*(st[active_bands]*\
                         (raw_data[:,passive_bands1]-A[passive_bands1])/st[passive_bands1]).T, axis=1)
    
    active_bands_2 = [a for a in active_bands if len(lattice[a])>1]
    passive_bands_2 = [lattice[b][1] for b in active_bands_2]
    second_sum2 = np.sum(weights*(st[active_bands_2]*\
                         (raw_data[:,passive_bands_2]-A[passive_bands_2])/st[passive_bands_2]).T, axis=1)

    two_neighbors = [i in active_bands_2 for i in active_bands]
    lsum = first_sum
    lsum[two_neighbors] += first_sum[two_neighbors]
    lsum -= second_sum1
    lsum[two_neighbors] -= second_sum2
    lsum[two_neighbors] /= 2
    lsum /= np.sum(weights)
    lsum[lsum > band_mins[active_bands]] = band_mins[active_bands][lsum > band_mins[active_bands]]
    lsum[lsum < 0] = 0
    
    return lsum


def atm_obj(data, weights):
    grads = data[:,1:]-data[:,:-1]
    full_obj = weights*(grads**2).T
    avg_obj = np.sum(full_obj)/np.sum(weights)
    return avg_obj


def atm_correction(cube, solar = 1.0, tol=0.01, verbose=False, approach_rate=1, est_min_R = 0.05, no_A=False):
    #generate mins
    mins = np.array([cube[:,i].min() for i in range(cube.shape[-1])])
    weights=1/np.sum((cube-mins+est_min_R)**2, axis=-1)
    objs = []
    
    #generate odd and even lattice
    bands = np.arange(len(mins))
    l1 = {}
    for i in bands[0::2]:
        connections = []
        if i-1 in bands:
            connections.append(i-1)
        if i+1 in bands:
            connections.append(i+1)
        l1[i] = connections
    l2 = {}
    for i in bands[1::2]:
        connections = []
        if i-1 in bands:
            connections.append(i-1)
        if i+1 in bands:
            connections.append(i+1)
        l2[i] = connections
    
    #initialize T, obj
    
    #A = mins*(1-est_min_R)
    T = (1-2*mins)
    standardization = np.ones_like(mins)
    med = int(len(mins)/2)
    for i in range(len(standardization)):
        # upstream typo kept on purpose ('standarization'): the triangular weighting below is
        # therefore never applied, as in MACHI ebbc3e72; reported upstream rather than changed here.
        standarization = np.maximum(0, 1-np.abs(i-med)/med)
    standardization /= standardization.max()
    A = mins*(1-est_min_R*standardization)
    if no_A:
        A*=0
    obj = atm_obj((cube-A)/(solar*T), weights)
    objs.append(obj)
    if verbose:
        print(obj)
    
    #run loop until tolerance is hit
    change = 1
    while change > tol:
        #update transmissions
        Tnew = T_update(cube, l1, A, T,
                  weights=weights, solar=solar)
        T[list(l1.keys())] = (1-approach_rate)*T[list(l1.keys())] + approach_rate*Tnew

        T[T > (1-2*A)] = 1-2*A[T > (1-2*A)]
        T[T<0] = 0
        Tnew = T_update(cube, l2, A, T, 
                  weights=weights, solar=solar)
        T[list(l2.keys())] = (1-approach_rate)*T[list(l2.keys())] + approach_rate*Tnew
        T[T > (1-2*A)] = 1-2*A[T > (1-2*A)]
        T[T<0] = 0
        
        #update scattering
        Anew = A_update(cube, l1, A, T, mins,
                  weights=weights, solar=solar)
        A[list(l1.keys())] = (1-approach_rate)*A[list(l1.keys())] + approach_rate*Anew
        Anew = A_update(cube, l2, A, T, mins,
                  weights=weights, solar=solar)
        A[list(l2.keys())] = (1-approach_rate)*A[list(l2.keys())] + approach_rate*Anew
        
        if no_A:
            A*=0
        obj_new = atm_obj((cube-A)/(solar*T), weights)
        change = (obj-obj_new)/obj
        obj = obj_new
        objs.append(obj)
        if verbose:
            print(obj)
            
    return T, A, objs
