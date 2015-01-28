"""Plots various measurements of CIV"""
import matplotlib.pyplot as plt
import numpy as np


#D'Odorico 2010 0910.2126 data
def plot_dor_omega_civ1():
    """Omega_CIV from D'Odorico 0910.2126. This is their Table 4 for N_CIV = 12 - 15"""
    omega_civ = [10.0, 5.0, 3.2, 2.4, 3.7]
    redshift = [1.75,2.25,2.75,3.25,3.75]
    #Bootstrap error
    omega_civ_err = [2.7, 1.4, 0.8, 0.6, 1.0]
    plt.errorbar(redshift, omega_civ, marker='o',fmt='none', yerr = omega_civ_err, xerr=0.25, ecolor="black")

def plot_dor_omega_civ2():
    """Omega_CIV from D'Odorico 0910.2126. This is their Table 4 for N_CIV = 13.8 - 15 (which is more directly comparable to Cooksey)"""
    omega_civ = [6.5, 4.4, 2.8,1.3]
    redshift = [1.8, 2.3, 2.75, 3.5]
    rederr = [0.3, 0.2, 0.25, 0.5]
    #Bootstrap error
    omega_civ_err = [1.9, 1.7, 0.9, 0.4]
    plt.errorbar(redshift, omega_civ, marker='o',fmt='none', yerr = omega_civ_err, xerr=rederr, ecolor="black")

def plot_dor_cddf(scale=1, moment=False):
    """CIV CDDF from D'Odorico 0910.2126. This is their Table 4 for N_CIV = 13.8 - 15 (which is more directly comparable to Cooksey)"""
    civ_cddf = np.array([[51,  12.18, -12.0708, 0.0608],
        [90,  12.48, -12.1242, 0.0458],
        [89,  12.78, -12.4290, 0.0460],
        [81,  13.08, -12.7699, 0.0483],
        [68,  13.38, -13.1459, 0.0527],
        [49,  13.68, -13.5882, 0.0620],
        [30,  13.98, -14.1013, 0.0793],
        #[18,  14.28, -14.6231, 0.1024],
        #[10,  14.58, -15.1784, 0.1373],
        #[6,  14.88, -15.7002, 0.1773 ],
    #[1,  15.18, -16.7784, 0.4343]
    ])
    lxer=-10**(civ_cddf[:,1]-0.15)+10**(civ_cddf[:,1])
    uxer=10**(civ_cddf[:,1]+0.15)-10**(civ_cddf[:,1])
    lyer=-10**(civ_cddf[:,2]-civ_cddf[:,3])+10**(civ_cddf[:,2])
    uyer=10**(civ_cddf[:,2]+civ_cddf[:,3])-10**(civ_cddf[:,2])
    #om_civ = np.trapz(10**civ_cddf[:,1]*10**civ_cddf[:,2],x=10**civ_cddf[:,1])
    #fact = (100/3.08567758e19)*1.67e-24*12/(1.88e-29*2.99e10)/0.72
    #print om_civ*fact
    if moment:
        scale = 10**civ_cddf[:,1]
    civ_cddf[:,2]+=np.log10(scale)
    lyer*=scale
    uyer*=scale
    plt.errorbar(10**civ_cddf[:,1], 10**civ_cddf[:,2], marker='o',fmt='none', yerr = [lyer, uyer], xerr=[lxer, uxer], ecolor="grey")

def plot_simcoe_data():
    """Plot the high redshift data from Simcoe 2011. 1104.4117 Integration limits are 13.4 - 15"""
    omega_civ = [1.87, 0.46]
    redshift = [4.95, 5.66]
    rederr = [[0.6,0.35], [0.36,0.74]]
    #Bootstrap error
    omega_civ_err = [0.5, 0.2]
    plt.errorbar(redshift, omega_civ, marker='o',fmt='none', yerr = omega_civ_err, xerr=rederr, ecolor="black")

#Cooksey 2013 1204.2827 data
def plot_c12_omega_civ():
    """Plot data from Cooksey 2012 CIV Omega"""
    redshift = [1.55687,  1.65971,  1.73999,  1.82360,  1.91460,  2.01778,  2.15320,  2.35608,  2.72298,  3.25860]
    rerr = np.array( [[1.46623, 1.60986],  [1.61004, 1.69993],  [1.70035, 1.78000],  [1.78007, 1.86985],  [1.87000, 1.95998],  [1.96002, 2.07997],  [2.08009, 2.23997],  [2.24015, 2.50914],  [2.51028, 2.96976],  [2.97005, 4.54334]])
    rederr = rerr.T - np.array([redshift,redshift])
    rederr[0,:]*=-1
    omega_civ =     [ 2.18, 2.10, 2.17, 2.11, 2.15, 1.86, 1.92, 1.61, 1.44, 0.87 ]
    omega_civ_err = [0.39,  0.32,  0.28,  0.23,  0.23,  0.17,  0.17,  0.18,  0.22,  0.13]
    plt.errorbar(redshift, omega_civ, marker='o',fmt='none', yerr = omega_civ_err, xerr=rederr, ecolor="black")

def plot_c12_line_den():
    """Plot data from Cooksey 2012 CIV line density"""
    redshift = [1.55687,  1.65971,  1.73999,  1.82360,  1.91460,  2.01778,  2.15320,  2.35608,  2.72298,  3.25860]
    rerr = np.array( [[1.46623, 1.60986],  [1.61004, 1.69993],  [1.70035, 1.78000],  [1.78007, 1.86985],  [1.87000, 1.95998],  [1.96002, 2.07997],  [2.08009, 2.23997],  [2.24015, 2.50914],  [2.51028, 2.96976],  [2.97005, 4.54334]])
    rederr = rerr.T - np.array([redshift,redshift])
    rederr[0,:]*=-1
    line_density = [ 0.332, 0.336, 0.356, 0.351, 0.355, 0.322, 0.312, 0.276, 0.221, 0.145 ]
    l_density_err = [0.011,  0.012,  0.013,  0.012,  0.013,  0.012,  0.012,  0.011,  0.009,  0.006]
    plt.errorbar(redshift, line_density, marker='o',fmt='none', yerr = l_density_err, xerr=rederr, ecolor="black")

def plot_c12_eqw_data(moment=False):
    """Plot the equivalent width histogram data from Cooksey 2012"""
    allf = np.loadtxt("fig_logfxw_rec.tab")
    #Each redshift has 15 entries. We want bin 6, at z=2.
    eqw = allf[5*15+1:6*15+2,:]
    mmebin = (eqw[:,0]+eqw[:,1])/2
    xer = mmebin-eqw[:,0]
    if moment:
        for i in (2,3,4):
            eqw[:,i]*=mmebin
    plt.errorbar(mmebin, eqw[:,2], marker='o',fmt='none', yerr = [eqw[:,3], eqw[:,4]], xerr=xer, ecolor="black")

def plot_c12_eqw_data_z35():
    """Plot the equivalent width histogram data from Cooksey 2012 at z=3.5"""
    allf = np.loadtxt("fig_logfxw_rec.tab")
    #Each redshift has 15 entries. We want the last (10th) bin, at z=3.5
    eqw = allf[9*15+2:10*15+2,:]
    mmebin = (eqw[:,0]+eqw[:,1])/2
    xer = mmebin-eqw[:,0]
    plt.errorbar(mmebin, eqw[:,2], marker='o',fmt='none', yerr = [eqw[:,3], eqw[:,4]], xerr=xer, ecolor="black")


