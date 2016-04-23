# -*- coding: utf-8 -*-
"""Module to plot statistics for the CIV absorbers around DLAs"""

from __future__ import print_function
import numpy as np
import plot_spectra as ps
import matplotlib.pyplot as plt
import laststar
import numexpr as ne
try:
    xrange(1)
except NameError:
    xrange = range

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


def _get_flux_weigh_vel(tau):
    """Compute the flux weighted velocity of a sightline"""
    vel = np.arange(np.size(tau))
    if np.sum(tau) == 0:
        mvel = 0.
    else:
        mvel = np.sum(vel*tau)/np.sum(tau)
    return mvel

class CIVPlot(ps.PlottingSpectra, laststar.LastStar):
    """Class to add some methods to PlottingSpectra which are useful for spectra around objects."""
    def get_offsets(self):
        """Get the offsets of each line in proper kpc from its partner"""
        midpoint = self.NumLos//2
        axisdiff = np.array([(self.cofm[ii,self.axis[ii]-1] - self.cofm[midpoint+ii,self.axis[ii]-1])**2 for ii in xrange(midpoint)])
        offsets = np.sqrt(np.sum((self.cofm[:midpoint,:] - self.cofm[midpoint:,:])**2,axis=1) - axisdiff)
        offsets *= self.atime/self.hubble
        assert np.size(offsets) == midpoint
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
            rtau1 = np.roll(t1, self.nbins/2-maxx)
            rtau2 = np.roll(t2, self.nbins/2-maxx)
            #Now compute eq. width for absorption +- N km/s from the center
            binwd = self.velsize/self.dvbin
            rtau1 = rtau1[self.nbins/2-binwd:self.nbins/2+binwd]
            rtau2 = rtau2[self.nbins/2-binwd:self.nbins/2+binwd]
            v1 = _get_flux_weigh_vel(rtau1)
            v2 = _get_flux_weigh_vel(rtau2)
            vdiff = v2 - v1
            if vdiff > self.nbins/2:
                vdiff = (vdiff-self.nbins/2)
            elif vdiff < - self.nbins/2:
                vdiff = (vdiff+self.nbins/2)
            offsets.append(vdiff*self.dvbin)
        vel_offsets = np.array(offsets)
        assert np.all(vel_offsets < 2*self.velsize)
        return vel_offsets

    def get_sum_col_density(self,elem, ion):
        """Get the column density summed over some (finite) velocity range around the deepest absorption"""
        cd = self.get_col_density(elem, ion)
        return self._get_sum_col_density(cd, elem, ion)

    def get_sum_collis_col_density(self, elem, ion):
        """Get the collisional column density summed over some (finite) velocity range around the deepest absorption"""
        cd = self.get_colis_colden(elem, ion)
        return self._get_sum_col_density(cd, elem, ion)

    def _get_sum_col_density(self,cd, elem, ion):
        """Get the column density summed over some (finite) velocity range around the deepest absorption"""
        cdd = self.get_col_density(elem, ion)
        sumcd = np.zeros(self.NumLos)
        midpoint = self.NumLos//2
        for i in xrange(midpoint):
            #First rotate lines so that the strongest absorber is in the center.
            cd1 = cdd[i, :]
            maxx = np.where(cd1 == np.max(cd1))[0][0]
            c1 = cd[i, :]
            c2 = cd[i+midpoint, :]
            rcd1 = np.roll(c1, self.nbins//2-maxx)
            rcd2 = np.roll(c2, self.nbins//2-maxx)
            binwd = self.velsize//self.dvbin
            #Now compute summed columns +- N km/s from the center
            sumcd[i] = np.sum(rcd1[self.nbins//2-binwd:self.nbins//2+binwd])
            sumcd[i+midpoint] = np.sum(rcd2[self.nbins//2-binwd:self.nbins//2+binwd])
        return sumcd

    def get_colden_par(self, elem, ion):
        """Get the column density along the sightline to the DLA"""
        #Get the spectra through the DLA
        midpoint = self.NumLos//2
        metal = self.get_col_density(elem,ion)[:midpoint]
        HI = self.get_col_density("H",1)[:midpoint]
        binwd = self.velsize//self.dvbin
        sumcd = np.zeros((midpoint, binwd))
        #Find the areas of each spectrum which are less than 600 kms from the DLA on either side
        #First find largest HI density
        for i in xrange(midpoint):
            his = HI[i,:]
            civs = metal[i,:]
            maxx = np.where(his == np.max(his))[0][0]
            rcivs = np.roll(civs, self.nbins//2-maxx)
            #Now compute summed columns +- N km/s from the center
            sumcd[i] = rcivs[self.nbins//2:self.nbins//2-binwd+1:-1] + rcivs[self.nbins//2:self.nbins//2+binwd]
        return sumcd

    def get_tau_par(self, elem, ion, line):
        """Get the column density along the sightline to the DLA"""
        #Get the spectra through the DLA
        midpoint = self.NumLos//2
        metal = self.get_tau(elem,ion, line)[:midpoint]
        HI = self.get_tau("H",1, 1215)[:midpoint]
        binwd = self.velsize//self.dvbin
        sumcd = np.zeros((midpoint, binwd))
        #Find the areas of each spectrum which are less than 600 kms from the DLA on either side
        #First find largest HI density
        for i in xrange(midpoint):
            his = HI[i,:]
            civs = metal[i,:]
            maxx = np.where(his == np.max(his))[0][0]
            rcivs = np.roll(civs, self.nbins//2-maxx)
            #Now compute summed columns +- N km/s from the center
            sumcd[i] = rcivs[self.nbins//2:self.nbins//2-binwd+1:-1] + rcivs[self.nbins//2:self.nbins//2+binwd]
        return sumcd

    def get_most_recent(self, elem="C", ion=4):
        """Get for every sightline the most recent enrichment event"""
        midpoint = self.NumLos//2
        colden = self.get_col_density(elem, ion)
        age = self.get_age(elem, ion)
        ages = []
        for (ag, cd) in zip(age[midpoint:, :], colden[midpoint:,:]):
            #Find the most recent event over all regions with significant CIV.
            #ind = np.where(cd > 1e12)
            #if np.size(ind) > 0:
                #ages.append(np.max(ag[ind]))
            #else:
            ages.append(ag[np.where(cd == np.max(cd))][0])
        return np.array(ages)

    def equivalent_width(self, elem, ion, line):
        """Calculate the equivalent width of a line in Angstroms. Algorithm (to match Rubin 2015) is:
            1. Start with +- 400 km/s (the mean HI interval of your spectra) on either side of the largest tau_HI
            within +-600 km/s of the strongest metal line in the DLA sightline.
            2. Extend asymmetrically in 100 km/s sections if mean(tau_HI) > 0.1 over that 100 km/s segment (which will pick up blatant DLAs).
            This gives me an HI interval similar to the thing you find by eye.
            3. Within this interval find the strongest metal line in the CGM sightline.
            4. Sum +-300 km/s from this line to get EW CIV. This number should make little difference: you set the limits smaller than the HI interval because there was no absorption there, so going for larger limits shouldn't change the EW much. Going too small though is not what you did, since you extended the region so that it covered the whole absorber. So we want something largeish here.
            5. Extend *lower only* (because of the doublet) in 100 km/s sections if mean(CIV) > 0.1 over that 100 km/s segment. This may not be worth the effort since 300-> 400 doesn't really make a difference.
            """
        tau = self.get_tau(elem, ion, line)
        tau_HI = self.get_tau("H", 1, 1215)
        eq_width = np.zeros(self.NumLos)
        midpoint = int(self.NumLos/2)
        lbinwd = int(self.velsize/self.dvbin)
        ubinwd = lbinwd
        midbin = int(self.nbins/2)
        #Doublets cannot be integrated past the doublet midpoint
        if line == 1548:
            ubinwd = int(250/self.dvbin)
        for i in xrange(midpoint):
            #First rotate lines so that the DLA is in the center.
            t1 = tau[i, :]
            t2 = tau[i+midpoint, :]
            #We use the metal line for the midpoint, as the DLA redshift is given by the metals
            maxx = np.where(t1 == np.max(t1))[0][0]
            rtau1 = np.roll(t1, midbin-maxx)
            rtau2 = np.roll(t2, midbin-maxx)
            #These are strong lines so it does not make much difference
            rtau1 = rtau1[midbin-lbinwd:midbin+ubinwd]
            #Now the tricky part: make HI centered on DLA
            rtauHI = np.roll(tau_HI[i+midpoint, :], midbin-maxx)
            #Now find the interval with 'significant HI absorption'
            maxion = self._get_strong_cgm_in_HI_vel(rtau2, rtauHI)
            rtau2 = np.roll(rtau2, midbin-maxion)
            #Now compute eq. width for absorption +- N km/s from the center
            rtau2 = rtau2[midbin-lbinwd:midbin+ubinwd]
            #1 bin in wavelength: δλ =  λ . v / c
            #λ here is the rest wavelength of the line.
            #speed of light in km /s
            light = self.units.light / 1e5
            #lambda in Angstroms, dvbin in km/s,
            #so dl is in Angstrom
            dl = self.dvbin / light * line
            eq_width[i] = np.trapz(-np.expm1(-rtau1),dx=dl)
            eq_width[i+midpoint] = np.trapz(-np.expm1(-rtau2),dx=dl)
        #Don't need to divide by 1+z as lambda_X is already rest wavelength
        assert np.any(eq_width > 0)
        return eq_width

    def _get_strong_cgm_in_HI_vel(self, rtauCGM, rtauHI):
        """Get the position of the strongest CGM line within a search interval centered on an HI absorber.
        Start with cgm tua centere on DLA"""
        #Now find the interval with 'significant HI absorption'
        (newcent, HIlbinwd, HIubinwd) = self._get_HI_vel_int(rtauHI)
        #Adjust CGM to match (this is so we don't get edge effects)
        rtauCGM = np.roll(rtauCGM, self.nbins//2-newcent)
        #Find the strongest metal line in this interval and center on it
        rtauCGMa = rtauCGM[self.nbins//2-HIlbinwd:self.nbins//2+HIubinwd]
        #Note the center is now at newcent, not self.nbins/2 because we need the unrotated frame
        maxion = np.where(rtauCGMa == np.max(rtauCGMa))[0][0]+newcent-HIlbinwd
        if maxion < 0:
            maxion += self.nbins
        if maxion > self.nbins:
            maxion -= self.nbins
        return maxion

    def _get_HI_vel_int(self, rtauHI):
        """Find the HI velocity interval and center from a single spectrum, centered on DLA"""
        #Find strongest HI in CGM sightline close to absorber
        initsearch = int(600/self.dvbin)
        midbin = int(self.nbins/2)
        irHI = rtauHI[midbin-initsearch:midbin+initsearch]
        newcent = np.where(irHI == np.max(irHI))[0][0]+midbin-initsearch
        #Adjust HI so this is now the center
        rtauHI2 = np.roll(rtauHI, midbin-newcent)
        #Now compute eq. width for absorption +- N km/s from the center
        #Extend the search radius if there is still strong HI absorption at the edges
        #First lower bound
        HIubinwd = int(round(400./self.dvbin))
        HIstep = int(round(100/self.dvbin))
        HIlbinwd = HIubinwd
        while np.mean(rtauHI2[midbin-(HIlbinwd + HIstep):midbin - HIlbinwd]) >= 0.1:
            HIlbinwd += HIstep
            if HIlbinwd+HIstep > midbin:
                HIlbinwd = midbin
                break
        #Then upper bound
        while np.mean(rtauHI2[midbin+HIubinwd:midbin + HIubinwd+HIstep]) >= 0.1:
            HIubinwd += HIstep
            if HIubinwd +HIstep >= midbin:
                HIubinwd = midbin-1
                break
        return (newcent, HIlbinwd, HIubinwd)

    def get_colis_colden(self, elem="C", ion=4):
        """Reload the collisionally ionised material fractions"""
        colden_collis = {}
        colden_collis[(elem,ion)] = np.array([0])
        self._really_load_array((elem, ion), colden_collis, "colden_collis")
        return colden_collis[(elem, ion)]

    def save_file(self):
        """Save a file including last time in star"""
        return laststar.LastStar.save_file(self)

    def load_savefile(self, savefile=None):
        laststar.LastStar.load_savefile(self,savefile)

    def find_nearest_halo(self, elem="H", ion=1, thresh=50):
        """Find the single most massive halos associated with absorption near a sightline, possibly via a subhalo."""
        try:
            return (self.spectra_halos, self.spectra_dists)
        except AttributeError:
            pass
#             try:
#                 self._load_spectra_halos()
#                 return (self.spectra_halos, self.spectra_dists)
#             except KeyError:
#                 pass
        self.load_halo()
        zpos = self.get_contiguous_regions(elem=elem, ion=ion, thresh = thresh)
        (halos, dists) = self.assign_to_halo(zpos=zpos, halo_radii=self.sub_radii, halo_cofm=self.sub_cofm)
        self.spectra_halos = halos
        self.spectra_dists = dists
#         self._save_spectra_halos()
        return (halos, dists)

    def assign_to_halo(self, zpos, halo_radii, halo_cofm, maxdist = 400):
        """
        Overload assigning positions to halos. As CIV absorbers are usually outside the virial radius,
        this is a bit complicated.

        We choose to find halos that observers would associate with the absorber. Observers will
        not be able to see subhalos, and will generally expect to find a bright galaxy within 100 kpc
        of the absorber.

        Thus look for the most massive halo within 170 kpc (physical), which is about a quasar virial radius.
        """
        dists = maxdist*np.ones_like(zpos)
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
            if np.all(lcolden == 0):
                contig[ii] = self.box/2
                continue
            maxx = np.where(np.max(lcolden) == lcolden)[0][0]
            low = (maxx - vrange//self.dvbin)
            high = (maxx + vrange//self.dvbin)
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


class AggCIVPlot(object):
    """Class to compute various statistics specific to absorbers around something else, like DLAs or quasars.
       Aggregates over a varied redshift range.
       First half of sightlines assumed to go through objects.
    """
    def __init__(self,nums, base, redfile, numlos=None, color=None, res=5., savefile="grid_spectra_DLA.hdf5",label="", ls="-", spec_res = 8.,load_halo=True, velsize = 300):
        #self.def_radial_bins = np.logspace(np.log10(7.5), np.log10(270), 12)
        #As is observed
        self.def_radial_bins = np.array([5,100,200,275])
        self.color=color
        self.label=label
        self.ls=ls
        self.velsize = velsize
        #Distribution of redshifts in the data
        self.datareds = np.loadtxt(redfile)[:,1]
        #Listify if necessary
        try:
            nums[0]
        except TypeError:
            nums = (nums,)
        nums = np.sort(nums)
        self.snaps = [CIVPlot(n, base, res=res, savefile=savefile, spec_res=spec_res,load_halo=load_halo) for n in nums]
        for ss in self.snaps:
            ss.velsize = velsize
        #Set the total number of sightlines as the number from the first snapshot
        #(ideally all snapshots have the same number), and all snapshots must have the same bin width
        if numlos == None:
            self.NumLos = self.snaps[0].NumLos
        else:
            self.NumLos = numlos
        self.nbins = self.snaps[0].nbins
        self.dvbin = self.snaps[0].dvbin
        #Seed for repeatability
        np.random.seed(52)
        (self.nlines, self.agg_map) = self._get_sample_map(self.NumLos/2)

    def equivalent_width(self, elem, ion, line):
        """Aggregated equivalent width"""
        eqw = [qq.equivalent_width(elem, ion, line) for qq in self.snaps]
        return self._get_aggregate(eqw)

    def _get_sample_map(self, npairs):
        """Get a map between sightlines in the underlying snapshots and sightlines in the aggregated structure.
        Does this by sampling (without replacement) the sightlines in the snapshots according to the observed redshift weights.
        We are sampling *pairs*, so the number of sightlines will be double npairs. """
        weights = self.get_redshift_weights()
        assert np.abs(np.sum(weights)-1.) < 1e-4
        npairs = int(npairs)
        nlines = np.floor(weights * npairs).astype(np.int)
        #Assign any extra to early bins
        leftover = npairs - np.sum(nlines)
        nlines[:leftover] += 1
        assert np.sum(nlines) == npairs
        #nlines now contains the number of sightlines we want from each snapshot
        if np.size(nlines > 1):
            print("Number of lines from each snapshot: ",[qq.red for qq in self.snaps],nlines*2)
        #Check that we have enough data to get this sample
        for i in xrange(np.size(self.snaps)):
            assert nlines[i] <= self.snaps[i].NumLos/2
        agg_map = [ np.random.choice(int(self.snaps[i].NumLos/2), size=nlines[i], replace=False) for i in xrange(np.size(nlines))]
        return (nlines, agg_map)

    def _get_aggregate(self, unaggregated, multiplier=2):
        """Get an aggregated quantity from a quantity passed as a list for the different snapshots
        according to the correct (observed) redshift distribution"""
        agg = np.empty(np.sum(self.nlines)*multiplier,dtype=unaggregated[0].dtype)
        total = 0
        for mm in xrange(multiplier):
            assert total == np.sum(self.nlines)*mm
            for jj in xrange(np.shape(unaggregated)[0]):
                agg[total:total+np.size(self.agg_map[jj])]= unaggregated[jj][mm*self.snaps[jj].NumLos//2+self.agg_map[jj]]
                total += np.size(self.agg_map[jj])
        assert total == np.size(agg)
        return agg

    def find_nearest_halo(self):
        """Find the nearest halo to the DLA sightline"""
        midpoint = self.NumLos/2
        near_halos = [qq.find_nearest_halo()[0] for qq in self.snaps]
        return self._get_aggregate(near_halos, multiplier=1)[:midpoint]

    def get_col_density(self, elem, ion):
        """Get the optical depths by aggregating over the different snapshots according to the correct density distribution"""
        cd = [qq.get_sum_col_density(elem, ion) for qq in self.snaps]
        return self._get_aggregate(cd)

    def get_col_density_par(self, elem, ion):
        """Get the column density parallel to the line of sight. Needs special handling because it has an odd shape."""
        cd = [qq.get_colden_par(elem, ion) for qq in self.snaps]
        agg = np.empty((np.sum(self.nlines), self.velsize//self.dvbin),dtype=cd[0].dtype)
        total = 0
        for jj in xrange(np.shape(cd)[0]):
            agg[total:total+np.size(self.agg_map[jj]),:]= cd[jj][self.agg_map[jj],:]
            total += np.size(self.agg_map[jj])
        assert total == np.shape(agg)[0]
        return agg

    def get_tau_par(self, elem, ion, line):
        """Get the column density parallel to the line of sight. Needs special handling because it has an odd shape."""
        cd = [qq.get_tau_par(elem, ion, line) for qq in self.snaps]
        agg = np.empty((np.sum(self.nlines), self.velsize//self.dvbin),dtype=cd[0].dtype)
        total = 0
        for jj in xrange(np.shape(cd)[0]):
            agg[total:total+np.size(self.agg_map[jj]),:]= cd[jj][self.agg_map[jj],:]
            total += np.size(self.agg_map[jj])
        assert total == np.shape(agg)[0]
        return agg

    def get_collis_col_density(self, elem, ion):
        """Get the optical depths by aggregating over the different snapshots according to the correct density distribution"""
        cd = [qq.get_sum_collis_col_density(elem, ion) for qq in self.snaps]
        return self._get_aggregate(cd)

    def get_offsets(self):
        """Get the optical depths by aggregating over the different snapshots according to the correct density distribution"""
        off = [qq.get_offsets() for qq in self.snaps]
        return self._get_aggregate(off,multiplier=1)

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
        if np.size(radial_bins) == 1 and radial_bins == None:
            radial_bins = self.def_radial_bins
        mean_plot_arr = np.zeros(np.size(radial_bins)-1)
        offsets = self.get_offsets()
        (center, mean_plot_arr) = _get_means_binned(plot_arr, offsets, radial_bins, mean)
        if label == None:
            label=self.label
        if line:
            plt.plot(center, mean_plot_arr, color=color, ls=ls, label=label)
        else:
            yerr = _generate_errors(plot_arr, offsets, radial_bins, np.size(offsets), 1000)
            #To count the number of labelled lines
            ax = plt.gca()
            _, l = ax.get_legend_handles_labels()
            center = np.array([(radial_bins[i]*(7-len(l))+(3+len(l))*radial_bins[i+1])/10. for i in range(0,np.size(radial_bins)-1)])
            plt.errorbar(center, mean_plot_arr, xerr=[center-radial_bins[1:],radial_bins[:-1]-center],yerr=yerr,fmt='s',color=color, label=label)
        return (center, mean_plot_arr)

    def plot_eq_width_ratio(self, color=None, ls="-", elem="C", ion=4, line=1548, radial_bins = None):
        """
        Compute a histogram of the ratios of equivalent widths in pairs of spectra.

        Returns:
            (v, f_table) - v (binned in log) and corresponding f(N)
        """
        if color == None:
            color=self.color
        eq_width = self.equivalent_width(elem, ion, line)
        midpoint = self.NumLos//2
        ratio = eq_width[midpoint:]/(eq_width[0:midpoint]+1e-5)
        return self._plot_radial(ratio, color, ls, None, radial_bins,mean=False)

    def plot_colden_ratio(self, color=None, ls="-", elem="C", ion=4, elem2=None, ion2=-1,radial_bins = None, label=None):
        """Column density plot; fraction of total in each ion"""
        if elem2 is None:
            elem2 = elem
        midpoint = self.NumLos/2
        totC = self.get_col_density(elem2,ion2)[midpoint:]
        CIV = self.get_col_density(elem,ion)[midpoint:]
        return self._plot_radial(CIV/(totC+1), color, ls, None, radial_bins, label=label,mean=True)

    def plot_colden(self, color=None, ls="-",ls2="--", elem="C", ion=4, radial_bins = None, label=None):
        """Column density plot"""
        midpoint = self.NumLos//2
        CIV = self.get_col_density(elem,ion)[midpoint:]
        return self._plot_radial(CIV, color, ls, ls2, radial_bins, label=label,mean=False)

    def plot_colden_par(self, color=None, ls="-",elem="C", ion=4, label=None):
        """Column density plot along an averaged DLA sightline"""
        #Get the spectra through the DLA
        CIV = self.get_col_density_par(elem,ion)
        #Find DLA position
        mean_civ = np.mean(CIV, axis=0)
        vbins = np.arange(0, self.velsize, self.dvbin)[:-1]
        if label == None:
            label=self.label
        assert np.size(vbins) == np.size(mean_civ)
        plt.plot(vbins, mean_civ, color=color, ls=ls, label=label)
        return (vbins, mean_civ)

    def plot_tau_par(self, color=None, ls="-",elem="C", ion=4, line=1548, label=None):
        """Column density plot along an averaged DLA sightline"""
        #Get the spectra through the DLA
        CIV = self.get_tau_par(elem,ion,line)
        #Find DLA position
        mean_civ = np.mean(CIV, axis=0)
        vbins = np.arange(0, self.velsize, self.dvbin)[:-1]
        if label == None:
            label=self.label
        assert np.size(vbins) == np.size(mean_civ)
        plt.plot(vbins, mean_civ, color=color, ls=ls, label=label)
        return (vbins, mean_civ)

    def plot_covering_fraction(self, eq_thresh = 0.2, color=None, ls="-", ls2 = "--", elem="C", ion=4, line=1548, radial_bins = None, label=None):
        """
        Plot the covering fraction of a given pair line above a threshold in radial bins
        """
        if color == None:
            color=self.color
        eq_width = self.equivalent_width(elem, ion, line)
        midpoint = self.NumLos//2
        assert np.shape(eq_width) == (self.NumLos,)
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
        midpoint = self.NumLos//2
        covering = np.zeros_like(cdensity[midpoint:])
        covering[np.where(cdensity[midpoint:] > cd_thresh)] = 1
        return self._plot_radial(covering, color, ls, "--", radial_bins, label=label,line=False)

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
        midpoint = self.NumLos//2  #self.nobs below
        yerr = _generate_errors(eq_width[:midpoint], np.zeros(midpoint), np.array([-1,1]), self.NumLos/2.,1000)
        plt.errorbar([0,], np.mean(eq_width[:midpoint]), yerr=yerr, color=color, fmt='s')
        return self._plot_radial(eq_width[midpoint:], color, ls, ls2, radial_bins, label=label,line=False)

    def get_vel_offset(self,elem, ion, line):
        """Get the velocity offset between a pair of sightlines"""
        vel_offsets = [qq.get_flux_vel_offset(elem, ion, line) for qq in self.snaps]
        vel_offset = self._get_aggregate(vel_offsets, multiplier=1)
        return vel_offset

    def plot_flux_vel_offset(self, eq_thresh = 0.2, color=None, elem="C", ion=4, line=1548):
        """
        Plot the covering fraction of a given pair line above a threshold in radial bins
        """
        if color == None:
            color = self.color
        vel_offset = self.get_vel_offset(elem, ion, line)
        eq_width = self.equivalent_width(elem, ion, line)
        midpoint = self.NumLos/2
        ind = np.where(eq_width[midpoint:] > eq_thresh)
#         lbins = np.linspace(0,300,16)
        lbins = np.linspace(0,120,7)
        (hist, _) = np.histogram(np.abs(vel_offset[ind]),lbins)
        norm = np.sum(hist)
        plt.bar(lbins[:-1], hist*12./norm, width=20, color=color, label=self.label, alpha=0.4)
        return (lbins, hist)

    def plot_mass_hist(self, dm=0.3, color=None, ls="-"):
        """
        Compute a histogram of the host halo mass of each DLA spectrum.

        Parameters:
            dm - bin spacing

        Returns:
            (mbins, pdf) - Mass (binned in log) and corresponding PDF.
        """
        if color == None:
            color = self.color
        halos = self.find_nearest_halo()
        f_ind = np.where(halos != -1)
        #nlos = np.shape(vel_width)[0]
        #print('nlos = ',nlos)
        mass = self.snaps[0].sub_mass[halos[f_ind]]
        m_table = 10**np.arange(np.log10(np.min(mass)+0.1), np.log10(np.max(mass)), dm)
        mbin = np.array([(m_table[i]+m_table[i+1])/2. for i in range(0,np.size(m_table)-1)])
        pdf = np.histogram(np.log10(mass),np.log10(m_table), density=True)[0]
        plt.semilogx(mbin, pdf, color=color, ls=ls, label=self.label)
        print("Field DLAs: ",np.size(halos)-np.size(f_ind))
        return (mbin, pdf)

    def plot_collisional_fraction(self, elem = "C", ion = 4, color=None, ls="-", label=None):
        """
        Plot fraction of absorbers that are collisionally ionised as a function of column density.
        """
        if label == None:
            label=self.label
        if color == None:
            color = self.color
        cdens = self.get_col_density(elem, ion)
        collis = self.get_collis_col_density(elem, ion)
        cratio = collis / (cdens+0.01)
        midpoint = self.NumLos/2  #self.nobs below
        new_radial_bins = np.logspace(np.log10(7.5), np.log10(270), 12)
        plt.plot(0,np.mean(cratio[:midpoint]),'o',color=color)
        return self._plot_radial(cratio[midpoint:], color, ls, 0, new_radial_bins, label=label,line=True)
        #plt.semilogx(center, fracs,ls=ls,color=color,label=label)
        #plt.xlabel(r"N$_\mathrm{CIV}$ (cm$^{-2}$)")
        #plt.ylim(0, 1)
        #print("Fraction with no halo: ",1-np.size(assoc)/1./np.size(halos))

