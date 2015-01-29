"""Make various plots of the total CIV abundance"""
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import numpy as np
import os.path as path
import myname
from civ_data import *
from civ_plotspectra import CIVPlottingSpectra
import line_data
from save_figure import save_figure

outdir = path.join(myname.base, "civ_plots/")

print "Plots at ",outdir

colors = {0:"pink", 1:"purple", 2:"cyan", 3:"green", 4:"gold", 5:"red", 7:"blue", 6:"grey", 8:"pink", 9:"orange", 'A':"grey", 'S':"red"}
lss = {0:"--",1:":", 2:":",3:"-.", 4:"--", 5:"-",6:"--",7:"-", 8:"-",9:"-",'A':"--"}
labels = {0:"ILLUS",1:"HVEL", 2:"HVNOAGN",3:"NOSN", 4:"WMNOAGN", 5:"MVEL",6:"METAL",7:"DEF", 8:"RICH",9:"FAST", 'A':"MOM", 'S':"SMALL"}

snaps = {1:4, 2:3.5, 3:3, 4:2.5, 5:2, 6:1.5, 7:1, 8:0.5, 9:0.3, 10:0, 901:6, 902:5, 903:4.5}
def plot_line_density(sim, box, end=6, early=False):
    """Plot the line density to compare with Cooksey 2012.
       Threshold is 0.6 A."""
    base = myname.get_name(sim, box=box)
    #Line density
    lciv = []
    #Omega_CIV > 14
    om_civ = []
    #Omega_CIV 12-15
    om_civ_low = []
    #Redshift
    reds = []
    sss = range(1,end)
    if early:
        sss = range(901,904)+sss
    for snap in sss:
        try:
            ahalo = CIVPlottingSpectra(snap, base, None, None, savefile="rand_civ_spectra.hdf5", spec_res=5.)
            reds.append(snaps[snap])
            lciv.append(ahalo.line_density_eq_w(0.6,"C",4,1548))
            om_civ.append(10**8*ahalo.omega_abs(10**(13.8),1e15,"C",4))
            om_civ_low.append(10**8*ahalo.omega_abs(1e12,1e15,"C",4))
        except IOError:
            #This snapshot doesn't exist
            continue
    plt.figure(1)
    plt.semilogy(reds, lciv, ls=lss[sim], color=colors[sim], label=labels[sim]+" "+str(box))
    plt.figure(2)
    plt.semilogy(reds, om_civ, ls=lss[sim], color=colors[sim], label=labels[sim]+" "+str(box))
    plt.figure(3)
    plt.semilogy(reds, om_civ_low, ls=lss[sim], color=colors[sim], label=labels[sim]+" "+str(box))

def plot_cddf(sim, snap, box):
    """Plot the CIV column density function"""
    base = myname.get_name(sim, box=box)
    #plt.figure(1)
    ahalo = CIVPlottingSpectra(snap, base, None, None, savefile="rand_civ_spectra.hdf5", spec_res=5.,label=labels[sim])
    #ahalo.plot_cddf("C", 4, minN=12, maxN=15., color=colors[sim], moment=False)
    #plt.figure(2)
    ahalo.plot_eq_width_dist("C",4,1548, color=colors[sim])

def linear_cog_col(eqw, rwave, fosc):
    """Plot the column density expected from the equivalent width, assuming we are in the linear regime of the curve of growth.
    ie: 1-e^(-tau) ~ tau
    This means that W ~ N and so we have (numerical values from Pettini Physical Cosmology notes):

    N   = 1.13x10^20  W / (lamdba^2 * fosc)
    """
    return 1.13e20 * eqw / (rwave**2 * fosc)

if __name__ == "__main__":
    sims = (1,2,3,4,7,9)
    ahalo = CIVPlottingSpectra(5, myname.get_name(7, box=25), None, None, savefile="rand_civ_spectra.hdf5", spec_res=5.,label=labels[0])
    ahalo.plot_eq_width_vs_col_den("C",4,1548)
    lines = line_data.LineData()
    fosc = lines[("C", 4)][1548].fosc_X
    eqw = np.linspace(-3, 0.5,50)
    plt.semilogy(eqw, linear_cog_col(10**eqw, 1548, fosc), '-',color="black")
    plt.ylim(1e11,1e17)
    plt.xlim(-3,0.5)
    save_figure(path.join(outdir,"civ_eqwvscolden"))
    plt.clf()
    ahalo.plot_eqw_mass("C",4,1548,color=colors[7])
    save_figure(path.join(outdir,"civ_eqwvsmass"))
    plt.clf()
    for s in sims:
        plot_cddf(s, 5, 25)
    #plt.figure(1)
    #plot_dor_cddf()
    #plt.legend(loc="upper right")
    #save_figure(path.join(outdir,"civ_cddf"))
    #plt.clf()
    #plt.figure(2)
    plt.xscale('log')
    plot_c12_eqw_data()
    plt.xlim(10**(-2.5), 5.0)
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    plot_dor_cddf(scale=1.13e20/1548**2/fosc)
    #plt.legend(loc="upper right")
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(1e-2,400)
    plt.xlim(linear_cog_col(10**(-2.5),1548, fosc), linear_cog_col(5.0,1548, fosc))
    save_figure(path.join(outdir,"civ_eqw"))
    plt.clf()
    for s in sims:
        try:
            plot_cddf(s, 2, 25)
        except IOError:
            pass
    #plt.figure(1)
    #plot_dor_cddf()
    #plt.legend(loc="upper right")
    #save_figure(path.join(outdir,"civ_cddf_z3.5"))
    #plt.clf()
    #plt.figure(2)
    plot_c12_eqw_data_z35()
    plt.xscale('log')
    #plt.legend(loc="upper right")
    plt.yscale('log')
    save_figure(path.join(outdir,"civ_eqw_z3.5"))
    plt.clf()
    for s in sims:
        plot_line_density(s, 25)
    #plot_line_density(4, 25,10)
    #plot_line_density(7, 25, early=True)
    #Small boxes seem too small. Alarming.
    #plot_line_density(5, 10)
    #plot_line_density(7, 7.5)
    plt.figure(1)
    plot_c12_line_den()
    plt.ylabel(r"$dN/dX \,(W_{r,1548} \geq W_{r, 0.6}$)")
    plt.xlabel("z")
    plt.xlim(1, 7)
    plt.ylim(1e-3, 1)
    plt.legend(loc="upper right")
    save_figure(path.join(outdir,"civ_line_dens"))
    plt.clf()
    plt.figure(2)
    plot_c12_omega_civ()
    plot_dor_omega_civ2()
    plot_simcoe_data()
    plt.ylabel(r"$\Omega_\mathrm{CIV} (10^{-8})$")
    plt.xlabel("z")
    plt.xlim(1, 9)
    plt.legend(loc="upper right")
    save_figure(path.join(outdir,"civ_omega"))
    plt.clf()
    plt.figure(3)
    plot_dor_omega_civ1()
    plt.ylabel(r"$\Omega_\mathrm{CIV} (10^{-8})$")
    plt.xlabel("z")
    plt.xlim(1, 7)
    plt.legend(loc="upper right")
    save_figure(path.join(outdir,"civ_omega_low"))
    plt.clf()
