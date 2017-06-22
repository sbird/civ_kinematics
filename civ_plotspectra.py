# -*- coding: utf-8 -*-
"""Subclass PlottingSpectra for the CGM, adding a few methods and overriding the halo assignment."""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os.path as path
import fake_spectra.plot_spectra as ps
import fake_spectra.line_data
import numexpr as ne
import h5py
from save_figure import save_figure
try:
    xrange(1)
except NameError:
    xrange = range

class CIVPlottingSpectra(ps.PlottingSpectra):
    def equivalent_width(self, elem, ion, line, limit=400.):
        """Calculate the equivalent width of a line in Angstroms, limited to 400 km/s on either side of the strongest absorber.
        Range will be extended if there is still absorbtion and truncated if there is a doublet."""
        tau = self.get_tau(elem, ion, line)
        eq_width = np.zeros(self.NumLos)
        for i in xrange(self.NumLos):
            #First rotate lines so that the DLA is in the center.
            t1 = tau[i, :]
            maxx = np.where(t1 == np.max(t1))[0][0]
            mdbn = int(self.nbins/2)
            rtau1 = np.roll(t1, mdbn-maxx)
            lbinwd = int(limit/self.dvbin)
            #Extend as needed
            while np.max(rtau1[mdbn-(lbinwd +int(100/self.dvbin)):mdbn - lbinwd]) >= 0.1:
                lbinwd += int(100/self.dvbin)
            #Doublet is cut off
            if line == 1548:
                ubinwd = int(250/self.dvbin)
            #Now compute eq. width for absorption +- N km/s from the center
            rtau1 = rtau1[mdbn-lbinwd:mdbn+ubinwd]
            #1 bin in wavelength: δλ =  λ . v / c
            #λ here is the rest wavelength of the line.
            #speed of light in km /s
            light = self.units.light / 1e5
            #lambda in Angstroms, dvbin in km/s,
            #so dl is in Angstrom
            dl = self.dvbin / light * line
            eq_width[i] = np.trapz(-np.expm1(-rtau1),dx=dl)
        #Don't need to divide by 1+z as lambda_X is already rest wavelength
        assert np.any(eq_width > 0)
        assert np.all(eq_width >= 0)
        return eq_width

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
        width = np.diff(W_table)
        dX=self.units.absorption_distance(self.box, self.red)
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
        return (W,f_W)

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

    def plot_eq_width_vs_col_den(self, elem, ion, line):
        """Plot the equivalent width vs the column density along the sightline for each spectrum."""
        eqw = np.log10(self.equivalent_width(elem, ion, line))
        colden = np.sum(self.get_col_density(elem, ion), axis=1)
        edges = np.linspace(np.min(eqw), np.max(eqw), 15)
        centers = (edges[1:]+edges[:-1])/2
        median_colden = [np.median(colden[np.where(np.logical_and(eqw > lbin, eqw < hbin))]) for (lbin, hbin) in zip(edges[:-1], edges[1:])]
        upquart = [np.percentile(colden[np.where(np.logical_and(eqw > lbin, eqw < hbin))],75) for (lbin, hbin) in zip(edges[:-1], edges[1:])]
        lquart = [np.percentile(colden[np.where(np.logical_and(eqw > lbin, eqw < hbin))],25) for (lbin, hbin) in zip(edges[:-1], edges[1:])]
        plt.semilogy(centers, median_colden,ls='--',color="brown",label=self.label)
        plt.semilogy(centers, upquart,ls=':',color="brown")
        plt.semilogy(centers, lquart,ls=':',color="brown")
        plt.xlabel(r"$W_{1548} (\AA)$")
        plt.ylabel(r"N$_\mathrm{CIV}$ (cm$^{2}$)")

    def assign_to_halo(self, zpos, halo_radii, halo_cofm, maxdist = 400):
        """
        Overload assigning positions to halos. As CIV absorbers are usually outside the virial radius,
        this is a bit complicated.

        We choose to find halos that observers would associate with the absorber. Observers will
        not be able to see subhalos, and will generally expect to find a bright galaxy within 100 kpc
        of the absorber.

        Thus look for the most massive halo within 170 kpc (physical), which is about a quasar virial radius.
        """
        dists = -1*np.ones_like(zpos)
        halos = np.zeros_like(zpos, dtype=np.int)-1
        #X axis first
        indd = np.where(self.sub_mass > 1e8)
        fhc = halo_cofm[indd,:][0]
        red_mass = self.sub_mass[indd]
        assert np.size(np.shape(fhc)) == 2
        assert np.shape(fhc)[1] == 3
        for ii in xrange(np.size(zpos)):
            proj_pos = np.array(self.cofm[ii,:])
            ax = self.axis[ii]-1
            proj_pos[ax] = zpos[ii]
            #Closest halo in units of halo virial radius
            #Find all halos within maxdist, accounting for periodicity
            box = self.box
            dd = ne.evaluate("sum(where(abs(fhc - proj_pos) < box/2., fhc-proj_pos, box-abs(fhc - proj_pos))**2, axis=1)")
            ind = np.where(dd < maxdist**2)
            if np.size(ind) > 0:
                max_mass = np.max(red_mass[ind])
                i2 = np.where(red_mass == max_mass)
                halos[ii] = indd[0][i2][0]
                dists[ii] = np.sqrt(dd[i2][0])
        return (halos, dists)

    def _save_spectra_halos(self):
        """Save the computed spectra and distances to the savefile."""
        f=h5py.File(self.savefile,'r+')
        try:
            del f["spectra"]["halos"]
            del f["spectra"]["dist"]
        except KeyError:
            pass
        f["spectra"]["halos"] = self.spectra_halos
        f["spectra"]["dist"] = self.spectra_dists
        f.close()

    def _load_spectra_halos(self):
        """Save the computed spectra and distances to the savefile."""
        f=h5py.File(self.savefile,'r')
        try:
            self.spectra_halos = np.array(f["spectra"]["halos"])
            self.spectra_dists = np.array(f["spectra"]["dist"])
        finally:
            f.close()

    def find_nearest_halo(self, elem="C", ion=4, thresh=50, maxdist=400):
        """Find the single most massive halos associated with absorption near a sightline, possibly via a subhalo."""
        try:
            return (self.spectra_halos, self.spectra_dists)
        except AttributeError:
            pass
            try:
                self._load_spectra_halos()
                return (self.spectra_halos, self.spectra_dists)
            except KeyError:
                pass
        zpos = self.get_contiguous_regions(elem=elem, ion=ion, thresh = thresh)
        (halos, dists) = self.assign_to_halo(zpos=zpos, halo_radii=self.sub_radii, halo_cofm=self.sub_cofm, maxdist=maxdist)
        self.spectra_halos = halos
        self.spectra_dists = dists
        self._save_spectra_halos()
        return (halos, dists)

    def plot_eqw_mass(self, elem = "C", ion = 4, line=1548, dlogW=0.5, minW=1e12, maxW=1e17, color="blue", maxdist=400):
        """Plot median halo mass for given equivalent width bins"""
        (halos,dists) = self.find_nearest_halo(elem,ion, thresh=50, maxdist=maxdist)
        eqw = np.sum(self.get_col_density(elem,ion),axis=1)
        W_table = 10**np.arange(np.log10(minW), np.log10(maxW), dlogW)
        center = np.array([(W_table[i]+W_table[i+1])/2. for i in xrange(0,np.size(W_table)-1)])
        width =  np.array([W_table[i+1]-W_table[i] for i in xrange(0,np.size(W_table)-1)])
        medians = np.ones_like(center)
        uquart= np.ones_like(center)
        lquart = np.ones_like(center)
        assoc = np.where(halos >=0)
        for ii in xrange(0,np.size(W_table)-1):
            #Lines in this bin
            ind = np.where(np.logical_and(eqw[assoc] < W_table[ii+1],eqw[assoc] > W_table[ii]))
            if np.size(ind) > 0:
                medians[ii] = np.median(self.sub_mass[halos[assoc][ind]])
                uquart[ii] = np.percentile(self.sub_mass[halos[assoc][ind]],75)
                lquart[ii] = np.percentile(self.sub_mass[halos[assoc][ind]],25)
        iii = np.where(medians > 2)
        plt.loglog(center[iii], medians[iii],ls="-",color=color,label=self.label)
        plt.loglog(center[iii], uquart[iii],ls=":",color=color)
        plt.loglog(center[iii], lquart[iii], ls=":",color=color)
        plt.ylabel(r"Mass ($M_\odot$ h$^{-1}$)")
        #plt.xlabel(r"$W_\mathrm{1548} (\AA )$")
        plt.xlabel(r"N$_\mathrm{CIV}$ (cm$^{-2}$)")
        plt.ylim(1e9,1e13)
        print("Fraction with no halo: ",1-np.size(assoc)/1./np.size(halos))

    def plot_eqw_dist(self, elem = "C", ion = 4, line=1548, dlogW=0.5, minW=1e12, maxW=1e17, color="blue", maxdist=400):
        """Plot median distance from halo in terms of virial radius as a function of column density (misnamed function!)"""
        (halos,dists) = self.find_nearest_halo(elem,ion, thresh=50,maxdist=maxdist)
        eqw = np.sum(self.get_col_density(elem,ion),axis=1)
        W_table = 10**np.arange(np.log10(minW), np.log10(maxW), dlogW)
        center = np.array([(W_table[i]+W_table[i+1])/2. for i in xrange(0,np.size(W_table)-1)])
        width =  np.array([W_table[i+1]-W_table[i] for i in xrange(0,np.size(W_table)-1)])
        medians = np.ones_like(center)
        uquart= np.ones_like(center)
        lquart = np.ones_like(center)
        assoc = np.where(halos >=0)
        for ii in xrange(0,np.size(W_table)-1):
            #Lines in this bin
            ind = np.where(np.logical_and(eqw[assoc] < W_table[ii+1],eqw[assoc] > W_table[ii]))
            if np.size(ind) > 0:
                medians[ii] = np.median(dists[assoc][ind])
                uquart[ii] = np.percentile(dists[assoc][ind],75)
                lquart[ii] = np.percentile(dists[assoc][ind],25)
        iii = np.where(medians > 2)
        plt.semilogx(center[iii],medians[iii], color=color,label=self.label)
        plt.semilogx(center[iii], uquart[iii],ls=":",color=color)
        plt.semilogx(center[iii], lquart[iii], ls=":",color=color)
        print("Fraction with no halo: ",1-np.size(assoc)/1./np.size(halos))
        plt.ylabel(r"Distance (kpc)")
        plt.xlabel(r"N$_\mathrm{CIV}$ (cm$^{-2}$)")
        plt.xlim(minW,maxW)

    def get_sum_col_density(self,elem, ion, limit=600.):
        """Get the column density summed over some (finite) velocity range around the deepest absorption"""
        cd = self.get_col_density(elem, ion)
        sumcd = np.zeros(self.NumLos)
        for i in xrange(self.NumLos):
            #First rotate lines so that the strongest absorber is in the center.
            c1 = cd[i, :]
            maxx = np.where(c1 == np.max(c1))[0][0]
            rcd1 = np.roll(c1, self.nbins//2-maxx)
            binwd = int(limit/self.dvbin)
            #Now compute summed columns +- N km/s from the center
            sumcd[i] = np.sum(rcd1[self.nbins//2-binwd:self.nbins//2+binwd])
        return sumcd

    def mass_hist(self, dm=0.3, nmin=None, elem="C", ion=4):
        """
        Compute a histogram of host halo masses

        Parameters:
            dm - bin spacing

        Returns:
            (mbins, pdf) - Mass (binned in log) and corresponding PDF.
        """
        (halos, _) = self.find_nearest_halo()
        if nmin == None:
            f_ind = np.where(halos != -1)
        else:
            f_ind = np.where(np.logical_and(halos != -1, self.get_sum_col_density(elem, ion,600) > nmin))
        #nlos = np.shape(vel_width)[0]
        #print('nlos = ',nlos)
        virial = self.sub_mass[halos[f_ind]]
        m_table = 10**np.arange(np.log10(np.min(virial)+0.1), np.log10(np.max(virial)), dm)
        mbin = np.array([(m_table[i]+m_table[i+1])/2. for i in range(0,np.size(m_table)-1)])
        pdf = np.histogram(np.log10(virial),np.log10(m_table), density=True)[0]
        #print("Field absorbers: ",np.size(halos)-np.size(f_ind))
        return (mbin, pdf)

    def plot_mass_hist(self, elem = "C", ion = 4, nmin=None, color="blue", ls="-", label=None):
        """Plot a histogram of the host halo masses for the absorber"""
        if label == None:
            label = self.label
        (mbins, pdf) = self.mass_hist(nmin=nmin, elem=elem, ion=ion)
        plt.semilogx(mbins,pdf,color=color, ls=ls,label=label)
        #plt.legend(loc=1,ncol=3)
        #plt.ylim(-0.03,2.8)
        #plt.xlim(10,400)
        plt.ylabel(r"Number density")
        plt.xlabel(r"Mass ($M_\odot$ h$^{-1}$)")
        #plt.xticks((10, 40, 100, 400), ("10","40","100","400"))
        #save_figure(path.join(topdir,"cosmo_halos_feedback_z"+str(zz)))
        #plt.clf()

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
            low = int(maxx - vrange/self.dvbin)
            high = int(maxx + vrange/self.dvbin)
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

    def get_coliss_colden(self, elem, ion):
        """Reload the collisionally ionised material fractions"""
        colden_collis = {}
        colden_collis[('C',4)] = np.array([0])
        self._really_load_array((elem, ion), colden_collis, "colden_collis")
        return colden_collis[(elem, ion)]

    def plot_collisional_fraction(self, elem = "C", ion = 4, dlogW=0.5, minW=1e12, maxW=1e17, color="blue", ls="-", label=None):
        """
        Plot fraction of absorbers that are collisionally ionised as a function of column density.
        """
        if label == None:
            label=self.label
        cdens = self.get_col_density(elem, ion)
        collis = self.get_coliss_colden(elem, ion)
        cdens = np.sum(cdens, axis=1)
        cratio = np.sum(collis, axis=1) / (cdens+0.01)
        W_table = 10**np.arange(np.log10(minW), np.log10(maxW), dlogW)
        center = np.array([(W_table[i]+W_table[i+1])/2. for i in xrange(0,np.size(W_table)-1)])
        fracs= np.zeros_like(center)
        for ii in xrange(0,np.size(W_table)-1):
            #Lines in this bin
            ind = np.where(np.logical_and(cdens < W_table[ii+1],cdens > W_table[ii]))
            if np.size(ind) > 0:
                fracs[ii] = np.mean(cratio[ind])
        print(np.mean(cratio))
        plt.semilogx(center, fracs,ls=ls,color=color,label=label)
        plt.ylabel(r"Fraction Collisionally Ionised")
        plt.xlabel(r"N$_\mathrm{CIV}$ (cm$^{-2}$)")
        plt.ylim(0, 1)
        #print("Fraction with no halo: ",1-np.size(assoc)/1./np.size(halos))
