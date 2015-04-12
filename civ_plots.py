# -*- coding: utf-8 -*-
"""Module to plot statistics for the CIV absorbers around DLAs"""

import numpy as np
import plot_spectra as ps
import matplotlib.pyplot as plt

def _get_means_binned(spectra, offsets, radial_bins,mean=True):
    """Get the means of some spectral quantity binned radially"""
    center = np.array([(radial_bins[i]+radial_bins[i+1])/2. for i in range(0,np.size(radial_bins)-1)])
    mean_plot_arr = np.zeros(np.size(radial_bins)-1)
    for ii in np.arange(np.size(radial_bins)-1):
        arr_bin = spectra[np.where(np.logical_and(offsets > radial_bins[ii], offsets < radial_bins[ii+1]))]
        if np.size(arr_bin) == 0:
            continue
        if mean:
            mean_plot_arr[ii] = np.mean(arr_bin)
        else:
            mean_plot_arr[ii] = np.median(arr_bin)
    return (center, mean_plot_arr)

def _bootstrap_sample(spectra, offsets, radial_bins, nsamples, error=0):
    """Generate a Monte Carlo error sample of some spectra."""
    # Generate some Monte Carlo samples where each element is perturbed by
    # a Gaussian, sigma given by error.
    index = np.random.random_integers(0, np.size(spectra)-1, nsamples)
    bootstrap = spectra[index]
    bootoffsets = offsets[index]
    if error > 0.:
        bootstrap += np.random.normal(0,error,size=nsamples)
    booted = _get_means_binned(bootstrap, bootoffsets, radial_bins, True)
    return booted

def _generate_errors(spectra, offsets, radial_bins, nset, nsamples, error=0):
    """Generate a Monte Carlo error sample of some spectra.
    Arguments:
        spectra - set of spectra to subsample
        radial_bins - set of radial bins to subsample
        nset - Size of each subsample
        nsamples - number of subsamples to generate
        error - If > 0, perturb each element of the subsample with a Gaussian of this sd.
    """
    #Generate a bunch of bootstrapped samples, each of size nset,  from our spectra
    sampled_mean = np.array([_bootstrap_sample(spectra, offsets, radial_bins, nset, error)[1] for _ in xrange(nsamples)])
    meds = np.median(sampled_mean,axis=0)
    lerr = meds - np.percentile(sampled_mean, 50-34, axis=0)
    uerr = np.percentile(sampled_mean, 50+34, axis=0) - meds
    return [lerr, uerr]

class CIVPlot(ps.PlottingSpectra):
    """Class to add some methods to PlottingSpectra which are useful for spectra around objects."""
    def get_offsets(self):
        """Get the offsets of each line in proper kpc from its partner"""
        midpoint = self.NumLos/2
        axx = set([0,1,2])
        offsets = np.empty(midpoint)
        for ii in np.arange(midpoint):
            ax = list(axx - set([self.axis[ii]]))
            offsets[ii] = np.sqrt(np.sum((self.cofm[ii,ax] - self.cofm[ii+midpoint,ax])**2))
        offsets *= self.atime/self.hubble
        return offsets

    def get_flux_vel_offset(self, elem="C", ion=4, line=1548):
        """
        Get the velocity offset between the flux in the host object and the flux in the nearby pair
        """
        midpoint = self.NumLos/2
        tau = self.get_tau(elem, ion, line)
        offsets = []
        for (t1, t2) in zip(tau[0:midpoint, :], tau[midpoint:,:]):
            #First rotate lines so that the DLA is in the center.
            maxx = np.where(t1 == np.max(t1))[0][0]
            rtau1 = np.roll(t1, maxx)
            rtau2 = np.roll(t2, maxx)
            v1 = self._get_flux_weigh_vel(rtau1)
            v2 = self._get_flux_weigh_vel(rtau2)
            vdiff = v2 - v1
            if vdiff > self.nbins/2:
                vdiff = (vdiff-self.nbins/2)
            elif vdiff < - self.nbins/2:
                vdiff = (vdiff+self.nbins/2)
            offsets.append(vdiff*self.dvbin)
        vel_offsets = np.array(offsets)
        return vel_offsets

    def _get_flux_weigh_vel(self, tau):
        """Compute the flux weighted velocity of a sightline"""
        vel = np.arange(self.nbins)
        mvel = np.sum(vel*tau)/np.sum(tau)
        return mvel

class AggCIVPlot(object):
    """Class to compute various statistics specific to absorbers around something else, like DLAs or quasars.
       Aggregates over a varied redshift range.
       First half of sightlines assumed to go through objects.
    """
    def __init__(self,nums, base, redfile, color=None, res=5., savefile="grid_spectra_DLA.hdf5",label="", ls="-", spec_res = 8.):
        #self.def_radial_bins = np.logspace(np.log10(7.5), np.log10(270), 12)
        #As is observed
        self.def_radial_bins = np.array([5,100,200,275])
        self.color=color
        self.label=label
        self.ls=ls
        #Distribution of redshifts in the data
        self.datareds = np.loadtxt(redfile)[:,1]
        nums = np.sort(nums)
        self.snaps = [CIVPlot(n, base, res=res, savefile=savefile, spec_res=spec_res) for n in nums]
        #Set the total number of sightlines as the number from the first snapshot
        #(ideally all snapshots have the same number), and all snapshots must have the same bin width
        self.NumLos = self.snaps[0].NumLos
        self.nbins = self.snaps[0].nbins
        self.dvbin = self.snaps[0].dvbin

    def equivalent_width(self, elem, ion, line):
        """Aggregated equivalent width"""
        eqw = [qq.equivalent_width(elem, ion, line) for qq in self.snaps]
        return self._get_aggregate(eqw)

    def _get_aggregate(self, unaggregated):
        """Get an aggregated quantity from a quantity passed as a list for the different snapshots
        according to the correct (observed) redshift distribution"""
        agg = np.empty_like(unaggregated[0])
        weights = self.get_redshift_weights()
        assert np.abs(np.sum(weights)-1.) < 1e-4
        weights *=np.shape(agg)[0]
        total = 0
        for jj in xrange(len(unaggregated)-1):
            agg[total:total+int(weights[jj])] = unaggregated[jj][total:total+int(weights[jj])]
            total+=int(weights[jj])
        agg[total:]= unaggregated[-1][total:]
        return agg

    def get_col_density(self, elem, ion):
        """Get the optical depths by aggregating over the different snapshots according to the correct density distribution"""
        cd = [np.sum(qq.get_col_density(elem, ion),axis=1) for qq in self.snaps]
        return self._get_aggregate(cd)

    def get_offsets(self):
        """Get the optical depths by aggregating over the different snapshots according to the correct density distribution"""
        off = [qq.get_offsets() for qq in self.snaps]
        return self._get_aggregate(off)

    def get_redshift_weights(self):
        """Get the redshift weightings for each snapshot"""
        reds = [qq.red for qq in self.snaps]
        #Compute the width of redshift bins
        redbins = [5.,]+ [(reds[i]+reds[i+1])/2. for i in xrange(np.size(reds)-1)]+[0.,]
        #Compute the weights within each redshift bin
        weights = np.array([np.size(np.where(np.logical_and(self.datareds < redbins[i], self.datareds > redbins[i+1]))) for i in xrange(np.size(reds))])/1./np.size(self.datareds)
        return weights

    def _plot_radial(self, plot_arr, color, ls, _, radial_bins, label=None,mean=True,line=True):
        """Helper function plotting a derived something as a function of radius"""
        if radial_bins == None:
            radial_bins = self.def_radial_bins
        center = np.array([(radial_bins[i]+radial_bins[i+1])/2. for i in range(0,np.size(radial_bins)-1)])
        mean_plot_arr = np.zeros(np.size(radial_bins)-1)
        offsets = self.get_offsets()
        (center, mean_plot_arr) = _get_means_binned(plot_arr, offsets, radial_bins, mean)
        if label == None:
            label=self.label
        if line:
            plt.plot(center, mean_plot_arr, color=color, ls=ls, label=label)
        else:
            yerr = _generate_errors(plot_arr, offsets, radial_bins, np.size(offsets), 5000)
            plt.errorbar(center, mean_plot_arr, xerr=[radial_bins[:-1]-center,center-radial_bins[1:]],yerr=yerr,fmt='s',color=color, label=label)
        return (center, mean_plot_arr)

    def plot_eq_width_ratio(self, color=None, ls="-", ls2="--", elem="C", ion=4, line=1548, radial_bins = None):
        """
        Compute a histogram of the ratios of equivalent widths in pairs of spectra.

        Returns:
            (v, f_table) - v (binned in log) and corresponding f(N)
        """
        if color == None:
            color=self.color
        eq_width = self.equivalent_width(elem, ion, line)
        midpoint = self.NumLos/2
        ratio = eq_width[midpoint:]/(eq_width[0:midpoint]+1e-5)
        return self._plot_radial(ratio, color, ls, ls2, radial_bins,mean=False)

    def plot_colden_ratio(self, color=None, ls="-",ls2="--", elem="C", ion=4, elem2=None, ion2=-1,radial_bins = None, label=None):
        """Column density plot; fraction of total in each ion"""
        if elem2 is None:
            elem2 = elem
        midpoint = self.NumLos/2
        totC = self.get_col_density(elem2,ion2)[midpoint:]
        CIV = self.get_col_density(elem,ion)[midpoint:]
        return self._plot_radial(CIV/(totC+1), color, ls, ls2, radial_bins, label=label,mean=True)

    def plot_colden(self, color=None, ls="-",ls2="--", elem="C", ion=4, radial_bins = None, label=None):
        """Column density plot"""
        midpoint = self.NumLos/2
        CIV = self.get_col_density(elem,ion)[midpoint:]
        return self._plot_radial(CIV, color, ls, ls2, radial_bins, label=label,mean=False)

    def plot_covering_fraction(self, eq_thresh = 0.2, color=None, ls="-", ls2 = "--", elem="C", ion=4, line=1548, radial_bins = None, label=None):
        """
        Plot the covering fraction of a given pair line above a threshold in radial bins
        """
        if color == None:
            color=self.color
        eq_width = self.equivalent_width(elem, ion, line)
        midpoint = self.NumLos/2
        covering = np.zeros_like(eq_width[midpoint:])
        covering[np.where(eq_width[midpoint:] > eq_thresh)] = 1
        #return self._plot_radial(covering, color, ls, ls2, self.obs_bins, label=label, line=False)
        return self._plot_radial(covering, color, ls, ls2, radial_bins, label=label, line=False)

    def plot_covering_fraction_colden(self, cd_thresh = 10**(17.2), color=None, ls="-", elem="H", ion=1, radial_bins = None, label=None):
        """
        Plot the covering fraction of a given pair line above a threshold in radial bins
        """
        if color == None:
            color=self.color
        cdensity = self.get_col_density(elem, ion)
        midpoint = self.NumLos/2
        covering = np.zeros_like(cdensity[midpoint:])
        covering[np.where(cdensity[midpoint:] > cd_thresh)] = 1
        return self._plot_radial(covering, color, ls, "--", radial_bins, label=label)

    def _get_errors(self, num, elem="C", ion=4):
        """Get errors on the equivalent widths by sampling from the observed errors"""
        if elem != "C" or ion != 4:
            return np.zeros(num, dtype=np.float64)
        sEWs = np.loadtxt("CGMofDLAs_Rperp.dat")[:,2]
        (hists, edges) = np.histogram(np.log10(sEWs))
        err = np.empty(num,dtype=np.float64)
        total = 0
        for ii in xrange(np.size(hists)):
            #How many sightlines in this bin?
            #The proportion from the observed survey, but at least 1 and no more
            #than we have left in the bag
            this = np.min((np.max((int(num*hists[ii]/1./np.sum(hists)),1)), num - total))
            err[total:total+this] = np.random.normal(loc=0.,scale=10**((edges[ii+1] + edges[ii])/2.),size=this)
            total+=this
        return err

    def plot_eq_width(self, color=None, ls="-", ls2 = "--", elem="C", ion=4, line=1548, radial_bins = None, label=None):
        """
        Plot the equivalent width of a given pair line above a threshold in radial bins
        """
        if color == None:
            color=self.color
#         eq_width = self.equivalent_width(elem, ion, line)+self._get_errors(self.NumLos, elem, ion)
        eq_width = self.equivalent_width(elem, ion, line)
        midpoint = self.NumLos/2  #self.nobs below
        yerr = _generate_errors(eq_width[:midpoint], np.zeros(midpoint), np.array([-1,1]), self.NumLos/2.,5000)
        plt.errorbar([0,], np.mean(eq_width[:midpoint]), xerr=[[0,],[7.5,]],yerr=yerr, color=color, fmt='s')
        return self._plot_radial(eq_width[midpoint:], color, ls, ls2, radial_bins, label=label,line=False)

    def plot_flux_vel_offset(self, eq_thresh = 0.2, color=None, elem="C", ion=4, line=1548):
        """
        Plot the covering fraction of a given pair line above a threshold in radial bins
        """
        if color == None:
            color = self.color
        vel_offsets = [qq.get_flux_vel_offset(elem, ion, line) for qq in self.snaps]
        vel_offset = self._get_aggregate(vel_offsets)
        eq_width = self.equivalent_width(elem, ion, line)
        midpoint = self.NumLos/2
        ind = np.where(eq_width[midpoint:] > eq_thresh)
        lbins = np.arange(0, 300, 20)
        (hist, _) = np.histogram(np.abs(vel_offset[ind]),lbins)
        norm = np.sum(hist)
        plt.bar(lbins[:-1], hist*12./norm, width=20, color=color, label=self.label, alpha=0.4)
        return (lbins, hist)

