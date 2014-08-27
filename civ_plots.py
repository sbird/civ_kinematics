"""Module to plot statistics for the CIV absorbers around DLAs"""

import numpy as np
import plot_spectra as ps
import matplotlib.pyplot as plt

#We will compute mean values in each radial bin

class CIVPlot(ps.PlottingSpectra):
    """Class to compute vrious statistics specific to the CIV near DLAs"""
    def_radial_bins = np.logspace(np.log10(25), np.log10(200), 10)
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

    def _plot_radial(self, plot_arr, color, ls, ls2, radial_bins):
        """Helper function plotting a derived something as a function of radius"""
        center = np.array([(radial_bins[i]+radial_bins[i+1])/2. for i in range(0,np.size(radial_bins)-1)])
        mean_plot_arr = np.zeros(np.size(radial_bins)-1)
        offsets = self.get_offsets()
        for ii in np.arange(np.size(radial_bins)-1):
            arr_bin = plot_arr[np.where(np.logical_and(offsets > radial_bins[ii], offsets < radial_bins[ii+1]))]
            if np.size(arr_bin) == 0:
                continue
            mean_plot_arr[ii] = np.mean(arr_bin)
        plt.plot(center, mean_plot_arr, color=color, ls=ls)
        return (center, mean_plot_arr)

    def plot_eq_width_ratio(self, color="blue", ls="-", ls2="--", elem="C", ion=4, line=1548, radial_bins = def_radial_bins):
        """
        Compute a histogram of the ratios of equivalent widths in pairs of spectra.

        Returns:
            (v, f_table) - v (binned in log) and corresponding f(N)
        """
        eq_width = self.equivalent_width(elem, ion, line)
        midpoint = self.NumLos/2
        ratio = eq_width[midpoint:]/eq_width[0:midpoint]
        return self._plot_radial(ratio, color, ls, ls2, radial_bins)

    def plot_covering_fraction(self, eq_thresh = 0.2, color="blue", ls="-", ls2 = "--", elem="C", ion=4, line=1548, radial_bins = def_radial_bins):
        """
        Plot the covering fraction of a given pair line above a threshold in radial bins
        """
        eq_width = self.equivalent_width(elem, ion, line)
        midpoint = self.NumLos/2
        covering = np.zeros_like(eq_width[midpoint:])
        covering[np.where(eq_width[midpoint:] > eq_thresh)] = 1
        return self._plot_radial(covering, color, ls, ls2, radial_bins)

    def plot_flux_vel_offset(self, eq_thresh = 0.2, color="blue", ls="-", ls2="--", elem="C", ion=4, line=1548, radial_bins = def_radial_bins):
        """
        Plot the covering fraction of a given pair line above a threshold in radial bins
        """
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
        plt.plot(center, mean_plot_arr, color=color, ls=ls)
        plt.plot(center, lower, color=color, ls=ls2)
        plt.plot(center, upper, color=color, ls=ls2)
        return (center, mean_plot_arr)

    def _get_flux_weigh_vel(self, tau):
        """Compute the flux weighted velocity of a sightline"""
        vel = np.arange(self.nbins)
        mvel = np.sum(vel*tau)/np.sum(tau)
        return mvel
