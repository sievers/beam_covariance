import numpy as np
from matplotlib import pyplot as plt
#from mpl_toolkits import mplot3d
import healpy
import pygsm
from scipy import interpolate
import time
from pygsm import GSMObserver
from pygsm import GlobalSkyModel
import read_edges
import read_highz
from datetime import datetime



def rotate_map(map,lat):
    myrot=healpy.rotator.Rotator([0,0,90-lat]);
    return myrot.rotate_map_pixel(map)
    #alm=healpy.map2alm(map)
    #healpy.rotate_alm(alm,theta=(lat*np.pi/180-np.pi/2))
    #return healpy.alm2map(alm,nside=healpy.npix2nside(len(map)))
                          
def almsdotalms(alms1,alms2,nside=None):    
    nalm=len(alms2)        
    lmax=np.int(np.sqrt(2*nalm))
    if nside is None:
        nside=np.int(lmax/3)
    if len(alms1.shape)==1:
        ans=2*np.dot(np.conj(alms1[lmax:]),alms2[lmax:])
        ans=ans+np.dot(alms1[:lmax],alms2[:lmax])
    else:
        ans=2*np.dot(np.conj(alms1[lmax:,:].T),alms2[lmax:])
        ans=ans+np.dot(alms1[:lmax,:].T,alms2[:lmax])
    return np.real(ans)*healpy.nside2npix(nside)/4/np.pi


def setup_edges():
    dd,freqs,th,phi=read_edges.read_edges()
    th=th*np.pi/180
    phi=phi*np.pi/180
    nspec=dd.shape[0]
    dat=np.zeros([dd.shape[0],dd.shape[2],dd.shape[1]])
    for i in range(nspec):
        dat[i,:,:]=np.fliplr(dd[i,:,:].T)
    return dat,freqs,th,phi

def setup_jm(tag):
    fname='./'+tag+'/results_pattern_' + tag + '_total90.dat'
    dat=np.loadtxt(fname,delimiter=',')
    f=open(fname,'r')
    ll=f.readline()[:-1]
    f.close()
    
    ii=ll.find(',')
    ll=ll[ii+2:]
    ii=ll.find(',')
    ll=ll[ii+2:]
    freqs=np.fromstring(ll,sep=',')[:-1]/1e6
        
    th=dat[:,0]
    phi=dat[:,1]
    
    nth=len(np.unique(th))
    nphi=len(np.unique(phi))

    th=th*np.pi/180
    phi=phi*np.pi/180

    thmat=np.reshape(th,[nphi,nth])
    phimat=np.reshape(phi,[nphi,nth])
    th=thmat[0,:]
    phi=phimat[:,0]
    
    dat=dat[:,2:]
    nspec=dat.shape[1]

    tmp=np.reshape(dat[:,0],[nphi,nth])
    dd=np.zeros([nspec,tmp.shape[0],tmp.shape[1]])
    for i in range(nspec):
        dd[i,:,:]=10**(np.reshape(dat[:,ii],[nphi,nth])/10)
    return dd,freqs,th,phi

def setup_highz(tag):
    beam,freqs,theta,phi=read_highz.read_dir(tag+'/')
    theta=theta*np.pi/180
    phi=phi*np.pi/180
    #for i in range(beam.shape[0]):
    #    beam[i,:,:]=np.fliplr(beam[i,:,:])
    return beam,freqs,theta,phi

plt.ion()




#tag='70mhz'  
#tag='100mhz';#dat,freqs,th,phi=setup_jm(tag)
#tag='MangoPeel'
#tag='EDGES';#dat,freq,th,phi=setup_edges()
tag='highz_20cm'

if tag=='EDGES':
    lat=-26
    print('Reading EDGES')
    #dat,freqs,th,phi=read_edges.read_edges()
    dat,freqs,th,phi=setup_edges()
    nspec=dat.shape[0]

if tag in ['70mhz','100mhz','MangoPeel']:
    lat=-47
    #this is all reading in the beam profiles and getting them into 2d arrays of theta,phi, which seems
    #to be pretty reliably what the beam simulation software produce
    print("Reading " + tag)
    dat,freqs,th,phi=setup_jm(tag)
    nspec=dat.shape[0]
if tag in ['highz_20cm']:
    lat=40.4
    print("Reading " + tag)
    dat,freqs,th,phi=setup_highz(tag)
    nspec=dat.shape[0]

#lat=lat*np.pi/180

#pick an nside, and set up healpix theta,phi values
nside=256
npix=healpy.nside2npix(nside)
hp_th,hp_phi=healpy.pix2ang(nside,np.arange(npix))
#sadly, scipy 2d interpolation forces us to do it in chunks of constant theta.  So, find the indices at which theta changes.
inds=np.where(np.diff(hp_th)!=0)[0]
inds=np.append(0,inds+1)

#hp_beam=myinterp(hp_phi,hp_th)

beams=np.zeros([len(hp_th),nspec])

tmp=healpy.sphtfunc.map2alm(beams[:,0])
nalm=len(tmp)
lvec=np.zeros(nalm,dtype='int64')
mvec=np.zeros(nalm,dtype='int64')
lmax=np.int(np.sqrt(2*nalm))

icur=0
for m in range(lmax):
    nn=lmax-m
    mvec[icur:icur+nn]=m
    lvec[icur:icur+nn]=np.arange(m,lmax)
    icur=icur+(lmax-m)


alms=np.zeros([nalm,nspec],dtype='complex')
t1=time.time()

beam_norm=np.zeros(nspec)

#loop over frequencies, then loop over theta values to create a set
#of beams interpolated onto healpix coordinates.  Then take their 
#spherical harmonic transforms.
for ii in range(nspec):
    #if tag=='EDGES':
    #    mat=dat[ii,:,:].copy()
    #    mat=np.fliplr(mat.T).copy()
    #else:
    #    mat=10**(np.reshape(dat[:,ii],[nphi,nth])/10)
    mat=dat[ii,:,:].copy()

    myinterp=interpolate.interp2d(th,phi,mat,'cubic',fill_value=0)
    hp_beam=np.zeros(len(hp_th))

    for i in range(np.int(len(inds)/2)+2):
        i1=inds[i]
        i2=inds[i+1]
        hp_beam[i1:i2]=myinterp(hp_th[i1],hp_phi[i1:i2])[:,0]
    beam_norm[ii]=np.sqrt(np.sum(hp_beam**2))
    #beams[:,ii]=rotate_map(hp_beam/np.sqrt(np.sum(hp_beam**2)),lat)
    beams[:,ii]=rotate_map(hp_beam/beam_norm[ii],lat)
    #beams[:,ii]=hp_beam/np.sqrt(np.sum(hp_beam**2))
    alms[:,ii]=healpy.sphtfunc.map2alm(beams[:,ii])
    #assert(1==0)
t2=time.time()
print('interpolated beam in ' + repr(t2-t1))


mycov=np.dot(beams.T,beams)
mycov2=2*np.dot(np.conj(alms[lmax:,:].T),alms[lmax:,:])+np.dot(alms[:lmax,:].T,alms[:lmax,:])
mycov2=mycov2*healpy.nside2npix(nside)/4/np.pi


plt.clf();plt.imshow(mycov,extent=[freqs.min(),freqs.max(),freqs.max(),freqs.min()])
plt.colorbar()
plt.title('Full Unweighted Correlation, ' + tag)
plt.xlabel('Freq (MHz)')
plt.ylabel('Freq (MHz)')
plt.savefig('corr_mat_nowt_' + tag + '.png')



cc=mycov[freqs<90,:];cc=cc[:,freqs<90]


#get the GSM, in this case evaluated at a single frequency (since this script was designed
#to investigate beam effects on the most boring possibly foregrounds)
gsm = GlobalSkyModel()
gsm_map_org=gsm.generate(80)
gsm_map_org=healpy.ud_grade(gsm_map_org,nside)

#if desired, cap the max flux in the map to be some number times the mean
if False:
    thresh=10
    mn=np.mean(gsm_map_org)
    gsm_map_org[gsm_map_org>thresh*mn]=thresh*mn
    gsm_tag='_capped'
else:
    gsm_tag=''

#rotate GSM into equatorial coordinates.
myrot=healpy.rotator.Rotator(coord=['G','E'])
gsm_map=myrot.rotate_map_pixel(gsm_map_org)
alms_gsm=healpy.map2alm(gsm_map)

#create a power spectrum model for the gsm
cl_gsm=healpy.sphtfunc.anafast(gsm_map_org)
lmin_fit=5
lmax_fit=100
pp=np.polyfit(np.log(np.arange(lmin_fit,lmax_fit)),np.log(cl_gsm[lmin_fit:lmax_fit]),1)
cl_gsm_fit=0*cl_gsm
cl_gsm_fit[0]=cl_gsm[0]
cl_gsm_fit[1:]=np.exp(pp[1])*(np.arange(1,len(cl_gsm))**pp[0])

for i in range(10):
    if cl_gsm_fit[i]>cl_gsm[i]:
        cl_gsm_fit[i]=cl_gsm[i]

plt.figure(1)
plt.clf();plt.loglog(cl_gsm);plt.loglog(cl_gsm_fit)
lims=list(plt.axis())
lims[1]=500
lims[3]=1e7
lims[2]=10
plt.axis(lims)
plt.legend(['GSM Spectrum','Power-law$^+$ fit'])
plt.xlabel('$l$')
plt.ylabel("$C_l$")
plt.title("GSM Spectrum vs. Power Law Fit")
plt.savefig('gsm_fit' + gsm_tag + '.png')

clvec=cl_gsm_fit[lvec]

alms_scaled=alms.copy()
for i in range(nspec):
    alms_scaled[:,i]=alms_scaled[:,i]*np.sqrt(clvec)


mycov_cl=np.real(2*np.dot(np.conj(alms_scaled[lmax:,:].T),alms_scaled[lmax:,:])+np.dot(alms_scaled[:lmax,:].T,alms_scaled[:lmax,:]))
tmp=np.diag(1.0/np.sqrt(np.diag(mycov_cl)))
mycov_cl=np.dot(tmp,np.dot(mycov_cl,tmp))
plt.clf();plt.imshow(mycov_cl,extent=[freqs.min(),freqs.max(),freqs.max(),freqs.min()])
plt.colorbar()
plt.title('Full $C_l$-weighted Correlation, ' + tag)
plt.xlabel('Freq (MHz)')
plt.ylabel('Freq (MHz)')
plt.savefig('corr_mat_cl_' + tag + '.png')

ee,vv=np.linalg.eig(mycov_cl)

if tag=='EDGES':
    numin=40
else:
    numin=55
numax=105
imin=np.min(np.where(freqs>numin))
imax=np.max(np.where(freqs<numax))
ee,vv=np.linalg.eig(mycov_cl[imin:imax,imin:imax])
print(np.sqrt(ee/ee[0])[:10])



beam_modes=np.dot(alms[:,imin:imax],vv)


#this is doing the beam times sky integration
#note that if we've expressed the beam in spherical harmonics,
#we can rotate it by an angle phi by multiplying by exp(i*m*phi)
#By orthogonality of Y_lm's, we can do the real-space integral
#by summing over l,m in the harmonic space.  Technically we should
#also do the negative m modes, but by symmetry that will just end up 
#doubling the real part/cancelling the imaginary, so we just
#take the real part



nha=96
ha=np.arange(nha)/(1.0*nha)*2*np.pi

amps=np.zeros([beam_modes.shape[1],nha])
for i in range(nha):
    amps[:,i]=almsdotalms(beam_modes,alms_gsm*np.exp(1J*ha[i]*mvec))

plt.figure(2)

plt.clf();
plt.plot(ha*12/np.pi,np.median(gsm_map)*amps[2,:]/np.mean(amps[0,:]))
plt.plot(ha*12/np.pi,np.median(gsm_map)*amps[3,:]/np.mean(amps[0,:]))
plt.xlabel("Hour Angle (arbitrary offset)")
plt.ylabel("Modes 2,3 Amps (K)")
plt.legend(['Mode 2','Mode 3'])
plt.title("GSM Amplitudes, " + tag)
plt.savefig("gsm_amps_" + tag + '.png')

plt.figure(6)
plt.clf()
plt.plot(ha*12/np.pi,np.median(gsm_map)*amps[0,:]/np.mean(amps[0,:]))
plt.xlabel("Hour Angle (arbitrary offset)")
plt.ylabel("Mode 1 Amps (K)")
plt.title("GSM Amplitudes, " + tag)
plt.savefig("gsm_amps_mode0_" + tag + '.png')



plt.clf()
mystd=np.std(amps,1)
plt.semilogy(mystd/mystd[0])
plt.plot(np.sqrt(ee/ee[0]),'*')
plt.legend(['GSM Mode Scatter','Gaussian Prediction'])
plt.xlabel('Mode Index')
plt.ylabel('$\sigma/\sigma[mode 0]$')
plt.title('Predicted vs. GSM $\sigma$, ' + tag)
plt.savefig('mode_sigmas_' + tag + '.png')

plt.figure(3)
leg=[]
n_to_plot=4
for i in range(n_to_plot):
    leg=leg+['Mode ' + repr(i)]

plt.clf();
plt.plot(freqs[imin:imax],vv[:,:n_to_plot])
plt.legend(leg)
plt.xlabel("MHz")
plt.ylabel("Mode Amplitude")
plt.title("Mode Frequency Behavior, " + tag)
plt.savefig('mode_freq_' + tag + '.png')


plt.clf();
tmp=vv[:,:n_to_plot].copy()
for i in range(n_to_plot):
    tmp[:,i]=tmp[:,i]*beam_norm[imin:imax]
plt.plot(freqs[imin:imax],tmp)
plt.legend(leg)
plt.xlabel("MHz")
plt.ylabel("Mode Amplitude Norm")
plt.title("Mode Frequency Behavior, " + tag)
plt.savefig('mode_freq_norm_' + tag + '.png')


plt.figure(4)
plt.clf()
for i in range(n_to_plot):
    tmp=healpy.alm2map(beam_modes[:,i].copy(),nside)
    plt.loglog(healpy.anafast(tmp))

plt.legend(leg)
plt.xlabel("$l$")
plt.ylabel("$C_l$")
plt.title("Mode Spectra, " + tag)
plt.savefig("mode_spectra_" + tag + '.png')




plt.figure(5)
nmode_use=5
time_corrs=np.zeros([nha,nmode_use])
time_eigs=np.zeros([nha,nmode_use])
for ii in range(nmode_use):
    beam_tmp=np.zeros([nalm,nha],dtype='complex')
    for i in range(nha):
        tmp=beam_modes[:,ii]*np.sqrt(clvec)
        beam_tmp[:,i]=tmp*np.exp(1J*ha[i]*mvec)
    time_corrs[:,ii]=np.real(almsdotalms(beam_tmp,tmp))
    mat=np.zeros([nha,nha])
    tmp=time_corrs[:,ii].copy()
    for i in range(nha):
        mat[:,i]=np.roll(tmp,i)
    mat=mat+mat.transpose()
    time_e,time_v=np.linalg.eig(mat)
    time_e=np.real(time_e)
    time_e.sort()
    time_eigs[:,ii]=np.real(time_e)
    r=np.linalg.cholesky(mat)
    nsim=10000;fwee=np.random.randn(r.shape[0],nsim);sim=np.dot(r,fwee);cc=np.dot(sim,sim.T)/nsim
    mn=np.mean(sim,axis=0);ss=np.std(sim,axis=0);print([ii,np.std(mn), np.mean(ss),np.sqrt(np.mean(sim**2))])


#plt.figure(5)
#nmode_use=5
#time_corrs=np.zeros([nha,nmode_use])
#time_eigs=np.zeros([nha,nmode_use])
#for ii in range(nmode_use):
#    beam_tmp=np.zeros([nalm,nha],dtype='complex')
#    for i in range(nha):
#        tmp=beam_modes[:,ii]*np.sqrt(clvec)
#        beam_tmp[:,i]=tmp*np.exp(1J*ha[i]*mvec)
#    time_corrs[:,ii]=np.real(almsdotalms(beam_tmp,tmp))
#    mat=np.zeros(nha)
#    for i in range(nha):
#        mat[:,i]=np.roll(time_corrs[:,ii],i)
#    time_e,time_v=np.linalg.eig(mat)
#    time_eigs[:,ii]=time_e
    

#(x, y, z, kind='linear', copy=True, bounds_error=False, fill_value=nan)



#vec=10**(dat[:,ii]/10)
#x=np.cos(th)*np.cos(phi)*vec
#y=np.cos(th)*np.sin(phi)*vec
#z=np.sin(th)*vec




#ax = plt.axes(projection='3d')
#ax.plot_trisurf(x, y, z) #, rstride=1, cstride=1,cmap='viridis', edgecolor='none',vmin=-0.5,vmax=6.0);

