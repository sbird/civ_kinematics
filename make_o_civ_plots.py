"""Make various plots of the total CIV abundance"""
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import numpy as np

labels = {1:"HVEL", 3:"NOSN", 7:"DEF", 9:"FAST", 4:"WARM"}
colors = {1:"purple", 3:"green", 7:"blue", 9:"red", 4:"gold"}

#Cooksey 2013 1204.2827 data
def plot_c12_omega_civ():
    """Plot data from Cooksey 2012 CIV Omega"""
    redshift = [1.96274,  1.55687,  1.65971,  1.73999,  1.82360,  1.91460,  2.01778,  2.15320,  2.35608,  2.72298,  3.25860]
    rerr = np.array( [[1.46623, 1.60986],  [1.61004, 1.69993],  [1.70035, 1.78000],  [1.78007, 1.86985],  [1.87000, 1.95998],  [1.96002, 2.07997],  [2.08009, 2.23997],  [2.24015, 2.50914],  [2.51028, 2.96976],  [2.97005, 4.54334]])
    rederr = rerr.T - np.array([redshift,redshift])
    rederr[0,:]*=-1
    omega_civ = [ 1.71, 2.18, 2.10, 2.17, 2.11, 2.15, 1.86, 1.92, 1.61, 1.44, 0.87 ]
    omega_civ_err = [0.20,  0.39,  0.32,  0.28,  0.23,  0.23,  0.17,  0.17,  0.18,  0.22,  0.13]
    plt.errorbar(redshift, omega_civ, marker='o',fmt='none', yerr = omega_civ_err, xerr=rederr)

def plot_c12_line_den():
    """Plot data from Cooksey 2012 CIV line density"""
    redshift = [1.55687,  1.65971,  1.73999,  1.82360,  1.91460,  2.01778,  2.15320,  2.35608,  2.72298,  3.25860]
    rerr = np.array( [[1.46623, 1.60986],  [1.61004, 1.69993],  [1.70035, 1.78000],  [1.78007, 1.86985],  [1.87000, 1.95998],  [1.96002, 2.07997],  [2.08009, 2.23997],  [2.24015, 2.50914],  [2.51028, 2.96976],  [2.97005, 4.54334]])
    rederr = rerr.T - np.array([redshift,redshift])
    rederr[0,:]*=-1
    line_density = [ 0.332, 0.336, 0.356, 0.351, 0.355, 0.322, 0.312, 0.276, 0.221, 0.145 ]
    l_density_err = [0.011,  0.012,  0.013,  0.012,  0.013,  0.012,  0.012,  0.011,  0.009,  0.006]
    plt.errorbar(redshift, line_density, marker='o',fmt='none', yerr = l_density_err, xerr=rederr)

snaps = {1:4, 2:3.5, 3:3, 4:2.5, 5:2}
def plot_line_density(sim, box):
    """Plot the line density to compare with Cooksey 2012.
    Threshold is 0.6 A."""
    base = myname.get_name(sim, box=box)
    lciv = []
    for snap in xrange(1,6):
        ahalo = ss.Spectra(snap, base, None, None, savefile="rand_civ_spectra.hdf5", spec_res=2.)
        reds.append(snaps[snap])
        lciv.append(ahalo.line_density_eq_w(0.6,"C",4,1548))
    plt.plot(reds, lciv, ls=lss[sim], color=colors[sim])
