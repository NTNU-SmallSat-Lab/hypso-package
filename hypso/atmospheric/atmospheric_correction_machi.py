'''
the Minimal Atmospheric Compensation for Hyperspectral Imagers

a very simple and somewhat fast atmospheric compensation that computes 
the transmission and scattering by optimizing the smoothness
of the recovered reflectances

jlgarrett 2024.07.08
joseph.garrett@ntnu.no
'''



import numpy as np
import matplotlib.pyplot as plt
import copy

def load_solar(wl):
    '''
    load the solar spectrum, interpolated to the wavelengths listed in wl
    '''
    solar_spectrum = np.loadtxt("hypso/atmospheric/thuillier_solar_spectrum.csv", skiprows=1, delimiter=';')
    solar_H1 = np.interp(wl, solar_spectrum[:,0], solar_spectrum[:,1] )
    return solar_H1


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


class AC:
    def __init__(self, A=0, T=1, solar=1):
        '''
        initializes as the identity if no parameters are input
        '''
        self.A = A
        self.T = T
        self.solar = solar
    
    def __call__(self, x):
        return (x-self.A)/(self.solar*self.T)
    
    def __repr__(self):
        return "mean transmission: {}".format(self.T.mean())
    
    def save(self, filename):
        '''
        should save as .npz
        '''
        if filename[:4] == '.npz':
            full_filename = filename
        else:
            print("adding saving as {}.npz".format(filename))
            full_filename = filename + '.npz'
        np.savez(full_filename, A=self.A, T=self.T, solar=self.solar)
    
    def load(self, filename):
        matrices = np.load(filename)
        self.T = matrices['T']
        self.A = matrices['A']
        self.solar = matrices['solar']
    
    def plot(self, wavelengths=False):
        fig, ax = plt.subplots(1,1)
        ax0 = ax.twinx()
        if wavelengths:
            ax.plot(wavelengths, T)
            ax0.plot(wavelengths, A, color='red')
        else:
            ax.plot(T)
            ax0.plot(A, color='red')
        
        ax0.set_ylabel("A")
        
            

def atm_correction(cube, solar = 1, tol=0.01, verbose=False):
    #generate mins
    mins = np.array([cube[:,i].min() for i in range(cube.shape[-1])])
    weights=1/np.sum(cube**2, axis=-1)
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
    T = np.ones_like(mins)
    A = copy.deepcopy(mins)
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
        T[list(l1.keys())] = Tnew

        Tnew = T_update(cube, l2, A, T, 
                  weights=weights, solar=solar)
        T[list(l2.keys())] = Tnew
        
        #update scattering
        Anew = A_update(cube, l1, A, T, mins,
                  weights=weights, solar=solar)
        A[list(l1.keys())] = Anew
        Anew = A_update(cube, l2, A, T, mins,
                  weights=weights, solar=solar)
        A[list(l2.keys())] = Anew
        
        
        obj_new = atm_obj((cube-A)/(solar*T), weights)
        change = (obj-obj_new)/obj
        obj = obj_new
        objs.append(obj)
        if verbose:
            print(obj)
            
    return T, A, objs



def run_machi(cube, solar = 1, tol=0.01, verbose=False):
    
    T, A, objs = atm_correction(cube=cube, solar=solar, tol=tol, verbose=verbose)

    return T, A, objs