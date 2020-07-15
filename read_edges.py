import numpy as np
#from matplotlib import pyplot as plt
#plt.ion()


def read_edges():
    fname='newniv.txt'
    data=np.genfromtxt(fname)
    f_original = np.arange(40,101,2)   
    beam_maps = np.zeros((len(f_original),91,360))
    
    for i in range(len(f_original)):
        beam_maps[i,:,:] = (10**(data[(i*360):((i+1)*360),2::]/10)).T
    th=np.linspace(0,90,beam_maps.shape[1])
    phi=np.linspace(0,360,beam_maps.shape[2])

    return beam_maps,f_original,th,phi
