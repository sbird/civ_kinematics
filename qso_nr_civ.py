# -*- coding: utf-8 -*-
"""Class to generate CIV spectra that are nearby to a DLA"""

import numpy as np
import math
import spectra
import sys
#import myname
import halocat
import hdfsim

class QSONrSpectra(spectra.Spectra):
    """Generate metal line spectra from simulation snapshot"""
    def __init__(self,num, base, numlos, min_mass, redmin, redmax, res = 5., cdir = None, savefile="nr_qso_spectra.hdf5", savedir=None, reload_file=True):
        #Load halos to push lines through them
        f = hdfsim.get_file(num, base, 0)
        self.OmegaM = f["Header"].attrs["Omega0"]
        self.box = f["Header"].attrs["BoxSize"]
        self.hubble = f["Header"].attrs["HubbleParam"]
        self.atime = f["Header"].attrs["Time"]
        self.redmin = redmin
        self.redmax = redmax
        self.npart=f["Header"].attrs["NumPart_Total"]+2**32*f["Header"].attrs["NumPart_Total_HighWord"]
        f.close()
        (_, self.sub_mass, cofm, self.sub_radii) = halocat.find_wanted_halos(num, base, self.hubble*min_mass/1e10)
        print np.size(self.sub_mass)," halos found"
        if np.size(self.sub_mass) == 0:
            raise ValueError
        self.sub_cofm = np.array(cofm, dtype=np.float64)
        self.NumLos = numlos
        #Re-seed for repeatability
        np.random.seed(23)
        (cofm, axis) = self.get_cofm(numlos)
        spectra.Spectra.__init__(self,num, base, None, axis, res, cdir, savefile=savefile,savedir=savedir,reload_file=reload_file)

    def get_cofm(self, num = None):
        """Find a bunch more sightlines: get a random sample of sightlines through quasars, then generate pairs offset by a random amount."""
        #All through x axis
        axis = np.ones(self.NumLos)
        select = np.random.randint(0,np.size(self.sub_mass),num)
        #Select the spectra
        cofm = np.concatenate((self.sub_cofm[select], self.sub_cofm[select]))
        axis = np.concatenate((axis, axis))
        #Add a small perturbation to the sightline cofm
        axx = set([0,1,2])
        rands = self.get_weighted_perp(num)
        #Note self.axis is 1 indexed, not 0 indexed
        for i in np.arange(num):
            ax = axx - set([axis[num+i]-1])
            cofm[num+i, list(ax)] += rands[i]
        return (cofm, axis)

    def get_weighted_perp(self, num):
        """Get a random perturbation with a radius weighted by the number of
           quasars in radial bins in the QPQ survey.
        """
        rperp = np.loadtxt("QSORperpred.txt")
        #Select quasars in the given redshift range
        ind = np.where(np.logical_and(rperp[:,1] > self.redmin, rperp[:,1] < self.redmax))
        rbins = np.logspace(1.5, 3, 15)
        (hists, rbins) = np.histogram(rperp[ind,0][0], rbins)
        conv = self.hubble/self.atime
        rbins *= conv
        phi = 2*math.pi*np.random.random_sample(num)
        rr = np.empty_like(phi)
        total = 0
        for ii in xrange(np.size(hists)-1):
            #How many sightlines in this bin?
            #The proportion from the observed survey, but at least 1 and no more
            #than we have left in the bag
            this = np.min((np.max((int(num*hists[ii]/1./np.sum(hists)),1)), num - total))
            rr[total:total+this] = (rbins[ii+1] - rbins[ii])*np.random.random_sample(this) + rbins[ii]
            total+=this
        this = num - total
        rr[total:] = (rbins[-1] - rbins[-2])*np.random.random_sample(this) + rbins[-2]
        assert np.max(rr) < rbins[-1]
        assert np.min(rr) > rbins[0]
        cofm=np.array([rr*np.cos(phi), rr*np.sin(phi)])
        assert np.shape(cofm) == (np.size(rr), 2)
        return cofm

def do_stuff(snap, path, redmin, redmax):
    """Make lines"""
    try:
        halo = QSONrSpectra(snap,path,5000, 10**12.5, redmin=redmin, redmax=redmax, savedir="/n/home11/spb/data/Illustris/snapdir_"+str(snap).rjust(3,'0'))
        halo.get_tau("C",4,1548, force_recompute=False)
        halo.get_tau("C",2,1334, force_recompute=False)
        halo.get_density("C",2)
        halo.get_density("C",4)
        halo.get_density("H",1)
        halo.save_file()
    except ValueError:
        print "No halos found, nothing done"

if __name__ == "__main__":
    #For small boxes
    #reds = {5:(1.,2.25), 4:(2.25, 2.75), 3:(2.75, 3.25)}
    #For big box (with better spacing)
    reds = {68:(1.,2.1), 66:(2.1, 2.25), 65:(2.25, 2.4465), 63:(2.4465,2.785), 60:(2.785,5.)}
    ibase = "/n/ghernquist/Illustris/Runs/Illustris-1/"
    j = int(sys.argv[1])
    zz = reds[j]
    do_stuff(j, ibase, zz[0], zz[1])

