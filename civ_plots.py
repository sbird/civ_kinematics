"""Module to plot statistics for the CIV absorbers around DLAs"""

import numpy as np
import plot_spectra as ps
import matplotlib.pyplot as plt

class CIVPlot(ps.PlottingSpectra):
    """Class to compute various statistics specific to the CIV near DLAs"""
    def_radial_bins = np.logspace(np.log10(7.5), np.log10(270), 12)
    obs_bins = np.array([5,100,200,275])
    color=None
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

    def _bootstrap_sample(self, spectra, offsets, radial_bins, nsamples, error=0):
        """Generate a Monte Carlo error sample of some spectra."""
        # Generate some Monte Carlo samples where each element is perturbed by
        # a Gaussian, sigma given by error.
        index = np.random.random_integers(0, np.size(spectra)-1, nsamples)
        bootstrap = spectra[index]
        bootoffsets = offsets[index]
        if error > 0.:
            bootstrap += np.random.normal(0,error,size=samples)
        booted = self._get_means_binned(bootstrap, bootoffsets, radial_bins, True)
        return booted

    def _generate_errors(self, spectra, offsets, radial_bins, nset, nsamples, error=0):
        """Generate a Monte Carlo error sample of some spectra.
        Arguments:
            spectra - set of spectra to subsample
            radial_bins - set of radial bins to subsample
            nset - Size of each subsample
            nsamples - number of subsamples to generate
            error - If > 0, perturb each element of the subsample with a Gaussian of this sd.
        """
        #Generate a bunch of bootstrapped samples, each of size nset,  from our spectra
        sampled_mean = np.array([self._bootstrap_sample(spectra, offsets, radial_bins, nset, error)[1] for i in xrange(nsamples)])
        meds = np.median(sampled_mean,axis=0)
        lerr = meds - np.percentile(sampled_mean, 50-34, axis=0)
        uerr = np.percentile(sampled_mean, 50+34, axis=0) - meds
        return [lerr, uerr]

    def _get_means_binned(self,spectra, offsets, radial_bins,mean=True):
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
        
    def _plot_radial(self, plot_arr, color, ls, ls2, radial_bins, label=None,mean=True,line=True):
        """Helper function plotting a derived something as a function of radius"""
        center = np.array([(radial_bins[i]+radial_bins[i+1])/2. for i in range(0,np.size(radial_bins)-1)])
        mean_plot_arr = np.zeros(np.size(radial_bins)-1)
        offsets = self.get_offsets()
        (center, mean_plot_arr) = self._get_means_binned(plot_arr, offsets, radial_bins, mean)
        if label == None:
            label=self.label
        if line:
            plt.plot(center, mean_plot_arr, color=color, ls=ls, label=label)
        else:
            yerr = self._generate_errors(plot_arr, offsets, radial_bins, np.size(offsets)/2., 5000)
            plt.errorbar(center, mean_plot_arr, xerr=[radial_bins[:-1]-center,center-radial_bins[1:]],yerr=yerr,fmt='s',color=color, label=label)
        return (center, mean_plot_arr)

    def plot_eq_width_ratio(self, color=None, ls="-", ls2="--", elem="C", ion=4, line=1548, radial_bins = def_radial_bins):
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

    def plot_colden_ratio(self, color=None, ls="-",ls2="--", elem="C", ion=4, elem2=None, ion2=-1,radial_bins = def_radial_bins, label=None):
        """Column density plot; fraction of total in each ion"""
        if elem2 is None:
            elem2 = elem
        midpoint = self.NumLos/2
        totC = np.sum(self.get_col_density(elem2,ion2),axis=1)[midpoint:]
        CIV = np.sum(self.get_col_density(elem,ion),axis=1)[midpoint:]
        return self._plot_radial(CIV/(totC+1), color, ls, ls2, radial_bins, label=label,mean=True)

    def plot_colden(self, color=None, ls="-",ls2="--", elem="C", ion=4, radial_bins = def_radial_bins, label=None):
        """Column density plot"""
        midpoint = self.NumLos/2
        CIV = np.sum(self.get_col_density(elem,ion),axis=1)[midpoint:]
        return self._plot_radial(CIV, color, ls, ls2, radial_bins, label=label,mean=False)

    def plot_covering_fraction(self, eq_thresh = 0.2, color=None, ls="-", ls2 = "--", elem="C", ion=4, line=1548, radial_bins = def_radial_bins, label=None):
        """
        Plot the covering fraction of a given pair line above a threshold in radial bins
        """
        if color == None:
            color=self.color
        eq_width = self.equivalent_width(elem, ion, line)
        midpoint = self.NumLos/2
        covering = np.zeros_like(eq_width[midpoint:])
        covering[np.where(eq_width[midpoint:] > eq_thresh)] = 1
        return self._plot_radial(covering, color, ls, ls2, self.obs_bins, label=label, line=False)
        #return self._plot_radial(covering, color, ls, ls2, radial_bins, label=label)

    def plot_covering_fraction_colden(self, cd_thresh = 10**(17.2), color=None, ls="-", elem="H", ion=1, radial_bins = def_radial_bins, label=None):
        """
        Plot the covering fraction of a given pair line above a threshold in radial bins
        """
        if color == None:
            color=self.color
        cdensity = np.sum(self.get_col_density(elem, ion), axis=1)
        midpoint = self.NumLos/2
        covering = np.zeros_like(cdensity[midpoint:])
        covering[np.where(cdensity[midpoint:] > cd_thresh)] = 1
        return self._plot_radial(covering, color, ls, "--", radial_bins, label=label)

    def plot_eq_width(self, color=None, ls="-", ls2 = "--", elem="C", ion=4, line=1548, radial_bins = def_radial_bins, label=None):
        """
        Plot the equivalent width of a given pair line above a threshold in radial bins
        """
        if color == None:
            color=self.color
        eq_width = self.equivalent_width(elem, ion, line)
        midpoint = self.NumLos/2
        yerr = self._generate_errors(eq_width[:midpoint], np.zeros(midpoint), np.array([-1,1]), midpoint/2., 5000)
        plt.errorbar([0,], np.mean(eq_width[:midpoint]), xerr=[[0,],[7.5,]],yerr=yerr, color=color, fmt='s')
        return self._plot_radial(eq_width[midpoint:], color, ls, ls2, self.obs_bins, label=label,line=False)

    def plot_flux_vel_offset(self, eq_thresh = 0.2, color=None, ls="-", ls2="--", elem="C", ion=4, line=1548, radial_bins = def_radial_bins):
        """
        Plot the covering fraction of a given pair line above a threshold in radial bins
        """
        if color == None:
            color=self.color
        midpoint = self.NumLos/2
        tau = self.get_tau(elem, ion, line)
        eq_width = self.equivalent_width(elem, ion, line)
        ind = np.where(eq_width[midpoint:] > eq_thresh)
        offsets = []
        for (t1, t2) in zip(tau[0:midpoint, :][ind], tau[midpoint:,:][ind]):
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
        center = np.array([(radial_bins[i]+radial_bins[i+1])/2. for i in range(0,np.size(radial_bins)-1)])
        mean_plot_arr = np.zeros(np.size(radial_bins)-1)
        upper = np.zeros(np.size(radial_bins)-1)
        lower = np.zeros(np.size(radial_bins)-1)
        offsets = self.get_offsets()[ind]
        for ii in np.arange(np.size(radial_bins)-1):
            arr_bin = vel_offsets[np.where(np.logical_and(offsets > radial_bins[ii], offsets < radial_bins[ii+1]))]
            if np.size(arr_bin) == 0:
                continue
            mean_plot_arr[ii] = np.median(arr_bin)
            upper[ii] = np.percentile(arr_bin,75)
            lower[ii] = np.percentile(arr_bin,25)
        plt.plot(center, mean_plot_arr, color=color, ls=ls, label=self.label)
        plt.plot(center, lower, color=color, ls=ls2)
        plt.plot(center, upper, color=color, ls=ls2)
        return (center, mean_plot_arr)

    def _get_flux_weigh_vel(self, tau):
        """Compute the flux weighted velocity of a sightline"""
        vel = np.arange(self.nbins)
        mvel = np.sum(vel*tau)/np.sum(tau)
        return mvel
