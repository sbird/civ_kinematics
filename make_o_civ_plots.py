"""Make various plots of the total CIV abundance"""
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import numpy as np
import os.path as path
import myname
import spectra as ss
from save_figure import save_figure

outdir = path.join(myname.base, "civ_plots/")

print "Plots at ",outdir

colors = {0:"pink", 1:"purple", 2:"cyan", 3:"green", 4:"gold", 5:"red", 7:"blue", 6:"grey", 8:"pink", 9:"orange", 'A':"grey", 'S':"red"}
lss = {0:"--",1:":", 2:":",3:"-.", 4:"--", 5:"-",6:"--",7:"-", 8:"-",9:"-",'A':"--"}
labels = {0:"ILLUS",1:"HVEL", 2:"HVNOAGN",3:"NOSN", 4:"WMNOAGN", 5:"MVEL",6:"METAL",7:"DEF", 8:"RICH",9:"FAST", 'A':"MOM", 'S':"SMALL"}

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

snaps = {1:4, 2:3.5, 3:3, 4:2.5, 5:2, 6:1.5, 7:1, 8:0.5, 9:0.3, 10:0, 901:6, 902:5, 903:4.5}
def plot_line_density(sim, box, end=6, early=False):
    """Plot the line density to compare with Cooksey 2012.
    Threshold is 0.6 A."""
    base = myname.get_name(sim, box=box)
    lciv = []
    om_civ = []
    reds = []
    if early:
        for snap in xrange(901,904):
            ahalo = ss.Spectra(snap, base, None, None, savefile="rand_civ_spectra.hdf5", spec_res=2.)
            reds.append(snaps[snap])
            lciv.append(ahalo.line_density_eq_w(0.6,"C",4,1548))
            om_civ.append(10**8*ahalo.omega_abs(10**14,10**22,"C",4))
    for snap in xrange(1,end):
        try:
            ahalo = ss.Spectra(snap, base, None, None, savefile="rand_civ_spectra.hdf5", spec_res=2.)
            reds.append(snaps[snap])
            lciv.append(ahalo.line_density_eq_w(0.6,"C",4,1548))
            om_civ.append(10**8*ahalo.omega_abs(10**14,10**22,"C",4))
        except IOError:
            #This snapshot doesn't exist
            continue
    plt.figure(1)
    plt.semilogy(reds, lciv, ls=lss[sim], color=colors[sim], label=labels[sim]+" "+str(box))
    plt.figure(2)
    plt.semilogy(reds, om_civ, ls=lss[sim], color=colors[sim], label=labels[sim]+" "+str(box))

if __name__ == "__main__":
    for s in (0,1,2,3,9):
        plot_line_density(s, 25)
    plot_line_density(4, 25,10)
    plot_line_density(7, 25, early=True)

    #Small boxes seem too small. Alarming.
    #plot_line_density(5, 10)
    #plot_line_density(7, 7.5)
    plt.figure(1)
    plot_c12_line_den()
    plt.ylabel(r"$dN/dX \,(W_{r,1548} \geq W_{r, 0.6}$)")
    plt.xlabel("z")
    plt.xlim(1, 7)
    plt.ylim(1e-3, 0.3)
    plt.legend(loc="upper right")
    save_figure(path.join(outdir,"civ_line_dens"))
    plt.clf()
    plt.figure(2)
    plot_c12_omega_civ()
    plt.ylabel(r"$\Omega_\mathrm{CIV} (10^{-8})$")
    plt.xlabel("z")
    plt.xlim(1, 7)

    plt.legend(loc="upper right")
    save_figure(path.join(outdir,"civ_omega"))
    plt.clf()

