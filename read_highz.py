import numpy as np
import glob

def read_freq(fname):
    f=open(fname,'r')
    ll=f.readlines()
    nl=len(ll)
    tags=ll[2].strip().split(',')
    #nc=len(tags)-1
    nc=len(tags)
    
    beam=np.zeros([nl-2,nc])
    for i in range(nl-2):
        line=ll[i+2]
        line=line.strip()
        #line=line[:-1]
        beam[i,:]=np.fromstring(line,sep=',')
    phi=beam[:,0].copy()
    beam=beam[:,1:].copy()
    line=ll[1].strip()[:-1]
    line=line[line.find(',')+1:]
    #print(line)
    th=np.fromstring(line,sep=',')
    return beam,th,phi


def read_dir(dirname):
    fnames=glob.glob(dirname+"/*.csv")
    #fnames.sort()
    nfreq=len(fnames)

    freqs=np.zeros(nfreq)
    for i in range(nfreq):
        tags=fnames[i].split('/')
        tags=tags[-1].split('M')
        freqs[i]=np.int(tags[0])
    inds=freqs.argsort()



    beam,th,phi=read_freq(fnames[inds[0]])
    beam_mat=np.zeros([nfreq,beam.shape[0],beam.shape[1]])
    beam_mat[0,:,:]=beam
    freqs[0]=freqs[inds[0]]
    for i in range(1,nfreq):
        fname=fnames[inds[i]]
        beam_mat[i,:,:],tt,pp=read_freq(fname)
        tags=fname.split('/')
        tags=tags[-1].split('M')
        freqs[i]=np.int(tags[0])

    return beam_mat,freqs,th,phi
