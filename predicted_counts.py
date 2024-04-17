import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaincc,expi,gamma

#constants
phi_star = 4.5e-3 #units of 1/MPc^3
alpha = -1.25
L_star = 7939.0 #Jy MPc^2
h = 0.7 #km/s/MPc (value from Papastergis Alfalfa paper)
om = 0.315 #omega matter
n_sigma = 5 #number of sigmas that I'm counting for a detection
Jy_to_SI = 1E-26
Pc_to_m = 3.085677581E16

################################################################ Cosmology/math functions stolen from Sievers
def get_t_of_a(a,h,Om):
    Ol=1.0-Om
    b=Om/Ol
    f=2*np.sqrt(a)*np.sqrt( (b+a**3)/a)*np.log(np.sqrt(b+a**3)+a**1.5)/np.sqrt(b+a**3)/3.0
    H=h*100*1e5/3.085678e24
    t=f/np.sqrt(Ol)/H
    return t

def get_comoving_vec(z0,npt=1000):
    z=np.linspace(0,z0,npt)
    a=1/(1+z)
    t=get_t_of_a(a,h,om)
    dt=np.abs(np.diff(t))
    aa=0.5*(a[1:]+a[:-1])
    dx=(dt/aa)/365.25/86400
    dx=dx/1e6/3.26156 #dx is now comoving differential length in Mpc
    xvec=np.cumsum(dx)
    xvec=np.append(0,xvec)
    xx=0.5*(xvec[1:]+xvec[:-1]);
    dv=4*np.pi*(xx)**2*(dx)
    vol=np.cumsum(dv)
    mydist=np.cumsum(dx)
    myz=z[1:]
    return mydist,vol,myz

def my_gammainc(alpha,x0):
    if alpha==-1:
        return -expi(-x0)
    if alpha>-1:
        return gammaincc(alpha+1,x0)*gamma(alpha+1)
    else:
        tmp=my_gammainc(alpha+1,x0)
        tmp2=-(x0**(alpha+1))*np.exp(-x0)
        return (tmp+tmp2)/(alpha+1)
    return None #never get here

def schech_int(L,alpha,L_star=1.0,phi_star=1.0):
    x=L/L_star
    y=my_gammainc(alpha,x)
    return y*phi_star

############################################################### stealing end

def skyFraction (z, D, theta):
    #the size of the primary beam changes with lambda
    #so really fsky should be a function of z also
    #not to mention the square degress of a rectangular strip depends on the declination
    #D is dish diameter in m
    #theta is declination in deg
    wavelength = (z+1)*0.21 #in m
    primaryBeamAngle = wavelength/D
    thetaLow = np.deg2rad(90-theta)-primaryBeamAngle
    thetaHigh = np.deg2rad(90-theta)+primaryBeamAngle
    totalSolidAngle = 2*np.pi*(np.cos(thetaLow)-np.cos(thetaHigh)) #in rad^2
    return totalSolidAngle/(4*np.pi)

def ncounts (zmax, SEFD, t_int, bandwidth, fsky):
    nbins = 25000
    #returns an array of dn/dz. SEFD in Jy, and t_int and bandwidth in the same units.
    D, vol, z = get_comoving_vec(zmax, npt=nbins)
    vol = np.append(0,vol) #for first bin
    L_min = (4*np.pi*D**2) * n_sigma*SEFD/np.sqrt(2*t_int*bandwidth)
    print("z_star:", z[np.argmin(np.abs(L_min-L_star))])
    n = schech_int(L_min, alpha, L_star, phi_star) #in units of number density per volume per bin
    n *= np.diff(vol) #counts per bin
    n = n/(zmax/nbins) #in units of dn/dz
    n *= fsky #accounting for sky fraction
    return z, n

if __name__ == "__main__":
    zmax = 0.7
    #some numbers for Alfalfa
    bw_alfalfa=1e8/4096 #Hz
    SEFD_alfalfa=(2.8+6*3.5)/7 #Jy
    alfalfa_tint = 48 #seconds
    fsky_alf=7000.0/(360.**2/np.pi)

    #some numbers for CHORD
    chord_footprint = 10000 #in deg
    fsky_chord=chord_footprint/(360**2/np.pi)
    chord_primary_beam_area = 12.0 #sq degrees
    chord_total_observing_time = 1000*24*3600 #sec
    chord_tint = chord_total_observing_time * (chord_primary_beam_area/chord_footprint)

    #for pathfinder
    pathfinder_tint = chord_tint
    fsky_pathfinder = fsky_chord/3.0

    z, chord_counts = ncounts(zmax, 12, chord_tint, bw_alfalfa, fsky_chord)
    print("Total CHORD count: ", int(np.trapz(chord_counts, dx=zmax/z.shape[0])))
    z, pathfinder_counts = ncounts(zmax, 94, pathfinder_tint, bw_alfalfa, fsky_pathfinder)
    print("Total Pathfinder count: ", int(np.trapz(pathfinder_counts, dx=zmax/z.shape[0])))
    z, alfalfa_counts = ncounts(zmax, SEFD_alfalfa, alfalfa_tint, bw_alfalfa, fsky_alf)
    print("Total Alfalfa count: ", int(np.trapz(alfalfa_counts, dx=zmax/z.shape[0])))
    plt.plot(z,chord_counts, label="CHORD")
    plt.plot(z,alfalfa_counts, label="Alfalfa")
    plt.plot(z,pathfinder_counts, label="Pathfinder")
    
    if False: #plot histogram of alfalfa counts
        table = np.loadtxt("alfalfa_table.csv", delimiter=",", skiprows=1, usecols=14)
        print("True alfalfa count:",table.shape)
        bin_edges, _, z = get_comoving_vec(zmax,100)
        bin_edges = np.append(0, bin_edges)
        histvalues, _ = np.histogram(table, bins=bin_edges)
        width = zmax/z.shape[0]
        plt.bar(z,histvalues/width,width=width,label="True Alfalfa", color="lightgrey")
    
    plt.gca().set_xlim([0,zmax])
    plt.legend()
    plt.title("Predicted number of counts",fontsize=20)
    plt.ylabel("dn/dz",fontsize=20)
    plt.xlabel("z",fontsize=20)
    plt.savefig("predicted_counts.png")
    plt.show()
    
