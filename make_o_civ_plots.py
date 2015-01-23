"""Make various plots of the total CIV abundance"""
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import numpy as np
import os.path as path
import myname
import spectra as ss
import plot_spectra as ps
from civ_data import *
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
            ahalo = ss.Spectra(snap, base, None, None, savefile="rand_civ_spectra.hdf5", spec_res=2.)
            reds.append(snaps[snap])
            lciv.append(ahalo.line_density_eq_w(0.6,"C",4,1548))
            om_civ.append(10**8*ahalo.omega_abs(10**14,10**22,"C",4))
            om_civ_low.append(10**8*ahalo.omega_abs(10**12,10**15,"C",4))
        except IOError:
            #This snapshot doesn't exist
            continue
    plt.figure(1)
    plt.semilogy(reds, lciv, ls=lss[sim], color=colors[sim], label=labels[sim]+" "+str(box))
    plt.figure(2)
    plt.semilogy(reds, om_civ, ls=lss[sim], color=colors[sim], label=labels[sim]+" "+str(box))
    plt.figure(3)
    plt.semilogy(reds, om_civ_low, ls=lss[sim], color=colors[sim], label=labels[sim]+" "+str(box))

class CIVPlottingSpectra(ps.PlottingSpectra):
    def eq_width_dist(self,elem = "C", ion = 4, line=1548, dW=0.05, minW=0.05, maxW=3.):
        """
        This computes the equivalent width frequency distribution, defined in analogy to the column density function.
        This is the number of detections per rest eq. width bin dW per co-moving path length dX.
        So we have f(N) = d n/ dW dX
        and n(N) = number of absorbers per sightline in this column density bin.
        Note f(N) has dimensions of 1/A, because N has units of cm^-2 and X is dimensionless.

        Returns:
            (W, f_W_table) - eq. width (linear binning) and corresponding f(W)
        """
        W_table = np.arange(minW, maxW, dW)
        center = np.array([(W_table[i]+W_table[i+1])/2. for i in range(0,np.size(W_table)-1)])
        dX=self.absorption_distance()
        #equivalent width for each sightline
        eqw = self.equivalent_width(elem, ion, line)
        tot_lines = np.size(eqw)
        (tot_f_W, W_table) = np.histogram(eqw,W_table)
        tot_f_W=tot_f_W/(dW*dX*tot_lines)
        return (center, tot_f_W)

    def plot_eq_width_dist(self,elem = "C", ion = 4, line=1548, dW=0.05, minW=0.05, maxW=3., color="blue"):
        """Plots the equivalent width frequency function."""
        (W,f_W)=self.eq_width_dist(elem, ion, line, dW,minW*0.9,maxW*1.1)
        plt.semilogy(W,f_W,color=color, label=self.label)
        ax=plt.gca()
        ax.set_xlabel(r"$W_{r,1548} (\AA)$")
        ax.set_ylabel(r"$f(W_{r,1548}) (\AA^{-1})$")
        plt.xlim(minW, maxW)

def plot_cddf(sim, box, snap=5):
    """Plot the CIV column density function"""
    base = myname.get_name(sim, box=box)
    plt.figure(1)
    ahalo = CIVPlottingSpectra(snap, base, None, None, savefile="rand_civ_spectra.hdf5", spec_res=2.,label=labels[sim])
    ahalo.plot_cddf("C", 4, minN=12, maxN=17., color=colors[sim])
    plt.figure(2)
    ahalo.plot_eq_width_dist("C",4,1548, color=colors[sim])

if __name__ == "__main__":
    ahalo = CIVPlottingSpectra(5, myname.get_name(0, box=25), None, None, savefile="rand_civ_spectra.hdf5", spec_res=2.,label=labels[0])
    ahalo.plot_eq_width_vs_col_den("C",4,1548)
    plt.ylim(1e11,1e17)
    plt.xlim(-3,0.5)
    save_figure(path.join(outdir,"civ_eqwvscolden"))
    plt.clf()
    for s in (0,1,2,3,9):
        plot_cddf(s, 25)
    plt.figure(1)
    plot_dor_cddf()
    plt.legend(loc="upper right")
    save_figure(path.join(outdir,"civ_cddf"))
    plt.clf()
    plt.figure(2)
    plot_c12_eqw_data()
    plt.legend(loc="upper right")
    plt.yscale('log')
    save_figure(path.join(outdir,"civ_eqw"))
    plt.clf()
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
