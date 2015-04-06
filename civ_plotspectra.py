"""Subclass PlottingSpectra for the CGM, adding a few methods and overriding the halo assignment."""
import matplotlib.pyplot as plt
import numpy as np
import os.path as path
import plot_spectra as ps
import line_data
import numexpr as ne
from save_figure import save_figure


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

        dX=ps.spectra.units.absorption_distance(self.box, self.red)
        #equivalent width for each sightline
        eqw = self.equivalent_width(elem, ion, line)
        tot_lines = np.size(eqw)+self.discarded
        (tot_f_W, W_table) = np.histogram(eqw,W_table)
        tot_f_W=tot_f_W/(width*dX*tot_lines)
        if moment:
            return (center, center*tot_f_W)
        return (center, tot_f_W)

    def plot_eq_width_dist(self,elem = "C", ion = 4, line=1548, dlogW=0.2, minW=0.004, maxW=5., color="blue", ls="-"):
        """Plots the equivalent width frequency function."""
        (W,f_W)=self.eq_width_dist(elem, ion, line, dlogW,minW*0.9,maxW*1.1)
        plt.semilogy(W,f_W,color=color, label=self.label, ls=ls)
        ax=plt.gca()
        ax.set_xlabel(r"$W_{1548} (\AA)$")
        ax.set_ylabel(r"$f(W_{1548}) (\AA^{-1})$")
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
            #Closest halo in units of halo virial radius
            dd = ne.evaluate("sum((halo_cofm - proj_pos)**2,axis=1)")
            indd = np.where(self.sub_mass > 0)
            dd[indd]/=halo_radii[indd]**2
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
        plt.ylabel(r"Mass ($M_\odot$ h$^{-1}$)")
        #plt.xlabel(r"$W_\mathrm{1548} (\AA )$")
        plt.xlabel(r"N$_\mathrm{CIV}$ (cm$^{-2}$)")
        plt.ylim(1e9,1e13)

    def plot_eqw_dist(self, elem = "C", ion = 4, line=1548, dlogW=0.5, minW=1e12, maxW=1e17, color="blue"):
        """Plot median distance from halo in terms of virial radius as a function of column density (misnamed function!)"""
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
                medians[ii] = np.median(dists[ind])
                uquart[ii] = np.percentile(dists[ind],75)
                lquart[ii] = np.percentile(dists[ind],25)
        plt.semilogx(center,medians, color=color)
        plt.semilogx(center, uquart,ls=":",color=color)
        plt.semilogx(center, lquart, ls=":",color=color)
        plt.ylabel(r"Distance (R$_\mathrm{vir}$)")
        plt.xlabel(r"N$_\mathrm{CIV}$ (cm$^{-2}$)")
        plt.xlim(minW,maxW)


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
        (roll, colden) = ps.spectra._get_rolled_spectra(den)
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
