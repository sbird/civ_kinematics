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
import line_data
import numexpr as ne
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

class CIVPlottingSpectra(ps.PlottingSpectra):
    def eq_width_dist(self,elem = "C", ion = 4, line=1548, dlogW=0.2, minW=0.005, maxW=5., moment=False):
        """
        This computes the equivalent width frequency distribution, defined in analogy to the column density function.
        This is the number of detections per rest eq. width bin dW per co-moving path length dX.
        So we have f(N) = d n/ dW dX
        and n(N) = number of absorbers per sightline in this column density bin.
        Note f(N) has dimensions of 1/A, because N has units of cm^-2 and X is dimensionless.

        Returns:
            (W, f_W_table) - eq. width (linear binning) and corresponding f(W)
        """
        W_table = 10**np.arange(np.log10(minW), np.log10(maxW), dlogW)
        #W_table = np.arange(minW, maxW, dW)
        center = np.array([(W_table[i]+W_table[i+1])/2. for i in range(0,np.size(W_table)-1)])
        width =  np.array([W_table[i+1]-W_table[i] for i in range(0,np.size(W_table)-1)])

        dX=self.absorption_distance()
        #equivalent width for each sightline
        eqw = self.equivalent_width(elem, ion, line)
        tot_lines = np.size(eqw)+self.discarded
        (tot_f_W, W_table) = np.histogram(eqw,W_table)
        tot_f_W=tot_f_W/(width*dX*tot_lines)
        if moment:
            return (center, center*tot_f_W)
        return (center, tot_f_W)

    def plot_eq_width_dist(self,elem = "C", ion = 4, line=1548, dlogW=0.2, minW=0.004, maxW=5., color="blue"):
        """Plots the equivalent width frequency function."""
        (W,f_W)=self.eq_width_dist(elem, ion, line, dlogW,minW*0.9,maxW*1.1)
        plt.semilogy(W,f_W,color=color, label=self.label)
        ax=plt.gca()
        ax.set_xlabel(r"$W_{r,1548} (\AA)$")
        ax.set_ylabel(r"$f(W_{r,1548}) (\AA^{-1})$")
        plt.xlim(minW, maxW)

    def line_density_dist(self, thresh=0.6, elem = "C", ion = 4, line=1548):
        """
        Compute dN/dX, by integrating dn/dWdX over W.
        """
        (bins, cddf) = self.eq_width_dist(elem, ion, line, minW = 0.6)
        #Integrate cddf * N
        #H0 in 1/s units
        h100=3.2407789e-18*self.hubble
        line_density = np.trapz(cddf, bins)
        return line_density

    def assign_to_halo(self, zpos, halo_radii, halo_cofm):
        """
        Overload assigning positions to halos. As CIV absorbers are usually outside the virial radius,
        this is a bit complicated.

        We choose to find halos that observers would associate with the absorber. Observers will
        not be able to see subhalos, and will generally expect to find a bright galaxy within 100 kpc
        of the absorber.

        We similarly abandon subhalos, and any concept of the virial radius (as it makes less sense for unbound objects).

        Instead we just look for the closest central halo.
        """
        dists = np.zeros_like(zpos)
        halos = np.zeros_like(zpos, dtype=np.int)
        #X axis first
        for ii in xrange(np.size(zpos)):
            proj_pos = np.array(self.cofm[ii,:])
            ax = self.axis[ii]-1
            proj_pos[ax] = zpos[ii]
            #Is this within the virial radius of any halo?
            dd = ne.evaluate("sum((halo_cofm - proj_pos)**2,axis=1)")
            indd = np.where(self.sub_mass > 0)
            #Minimal distance
            ind = np.where(dd[indd] == np.min(dd[indd]))
            halos[ii] = indd[0][ind][0]
            dists[ii] = np.sqrt(dd[indd][ind][0])
        return (halos, dists)

    def find_nearest_halo(self, elem="C", ion=4, thresh=50):
        """Find the single most massive halos associated with absorption near a sightline, possibly via a subhalo."""
        try:
            return (self.spectra_halos, self.spectra_dists)
        except AttributeError:
            pass
        zpos = self.get_contiguous_regions(elem=elem, ion=ion, thresh = thresh)
        (halos, dists) = self.assign_to_halo(zpos, self.sub_radii, self.sub_cofm)
        self.spectra_halos = halos
        self.spectra_dists = dists
        return (halos, dists)

    def plot_eqw_mass(self, elem = "C", ion = 4, line=1548, dlogW=0.5, minW=1e12, maxW=1e17, color="blue"):
        """Plot median halo mass for given equivalent width bins"""
        (halos,dists) = self.find_nearest_halo(elem,ion, 50)
        eqw = np.sum(self.get_col_density(elem,ion),axis=1)
        W_table = 10**np.arange(np.log10(minW), np.log10(maxW), dlogW)
        center = np.array([(W_table[i]+W_table[i+1])/2. for i in xrange(0,np.size(W_table)-1)])
        width =  np.array([W_table[i+1]-W_table[i] for i in xrange(0,np.size(W_table)-1)])
        medians = np.ones_like(center)
        uquart= np.ones_like(center)
        lquart = np.ones_like(center)
        for ii in xrange(0,np.size(W_table)-1):
            #Lines in this bin
            ind = np.where(np.logical_and(eqw < W_table[ii+1],eqw > W_table[ii]))
            if np.size(ind) > 0:
                medians[ii] = np.median(self.sub_mass[halos[ind]])
                uquart[ii] = np.percentile(self.sub_mass[halos[ind]],75)
                lquart[ii] = np.percentile(self.sub_mass[halos[ind]],25)
        plt.loglog(center, medians,color=color)
        plt.loglog(center, uquart,ls=":",color=color)
        plt.loglog(center, lquart, ls=":",color=color)
        plt.ylim(1e9,1e13)

    def get_contiguous_regions(self, elem="C", ion = 4, thresh = 50, relthresh = 1e-3):
        """
        Find the weighted z position of all CIV elements in a spectrum.
        Here we want 50 km/s +- the deepest absorption.
        Returns a list of lists. Each element in the outer list corresponds to a spectrum.
        Each inner list is the list of weighted z positions of regions.
        In this case the inner list will always have 1 element.
        """
        #Overload the thresh argument to actually be a velocity range
        vrange = thresh
        den = self.get_col_density(elem, ion)
        contig = np.zeros(self.NumLos,dtype=np.float)
        (roll, colden) = self._get_rolled_spectra(den)
        #deal with periodicity by making sure the deepest point is in the middle
        for ii in xrange(self.NumLos):
            # This is column density, not absorption, so we cannot
            # use the line width to find the peak region.
            lcolden = colden[ii,:]
            maxx = np.where(np.max(lcolden) == lcolden)[0][0]
            low = (maxx - vrange/self.dvbin)
            high = (maxx + vrange/self.dvbin)
            # Find weighted z position for absorber
            nn = np.arange(self.nbins)[low:high]-roll[ii]
            llcolden = lcolden[low:high]
            zpos = ne.evaluate("sum(llcolden*nn)")
            summ = ne.evaluate("sum(llcolden)")
            #Make sure it refers to a valid position
            zpos = (zpos / summ) % self.nbins
            zpos *= 1.*self.box/self.nbins
            contig[ii] = zpos
        return contig

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
