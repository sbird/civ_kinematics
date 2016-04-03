# -*- coding: utf-8 -*-
"""Class to generate CIV spectra that are nearby to a DLA"""

from __future__ import print_function
import numpy as np
import math
import spectra
import laststar
import myname
import hdfsim

class DLANrSpectra(laststar.LastStar):
    """Generate metal line spectra from simulation snapshot"""
    def __init__(self,num, base, numlos, redmin, redmax, res = 5., cdir = None, savefile="nr_dla_spectra.hdf5", savedir=None, reload_file=True):
        #Get a sample of DLAs from the savefile
        dlas = spectra.Spectra(num, base, None, None, res, cdir, savefile="grid_spectra_DLA.hdf5",savedir=savedir)
        dla_cofm = dlas.cofm
        dla_axis = dlas.axis
        self.redmin = redmin
        self.redmax = redmax
        f = hdfsim.get_file(num, base, 0)
        self.OmegaM = f["Header"].attrs["Omega0"]
        self.box = f["Header"].attrs["BoxSize"]
        self.hubble = f["Header"].attrs["HubbleParam"]
        self.atime = f["Header"].attrs["Time"]
        f.close()
        self.NumLos = numlos
        #Sightlines at random positions
        #Re-seed for repeatability
        np.random.seed(23)
        select = np.random.randint(0,dlas.NumLos,numlos)
        cofm = np.concatenate((dla_cofm[select], dla_cofm[select]))
        axis = np.concatenate((dla_axis[select], dla_axis[select]))
        #Add a small perturbation to the sightline cofm
        axx = set([0,1,2])
        rands = self.get_weighted_perp(numlos)
        for i in np.arange(0,numlos):
            ax = axx - set([axis[numlos+i]])
            cofm[numlos+i, list(ax)] += rands[i,:]
        spectra.Spectra.__init__(self,num, base, cofm, axis, res, cdir, savefile=savefile,savedir=savedir,reload_file=reload_file)
        self.age = {}

    def get_cofm(self, num = None):
        """Find a bunch more sightlines: do nothing in this case"""
        return num

    def get_weighted_perp(self, num):
        """Get a random perturbation with a radius weighted by the number of
           quasars in radial bins in the DLA-CGM survey."""
        rperp = np.loadtxt("DLARperpred.txt")
        #Select quasars in the given redshift range
        ind = np.where(np.logical_and(rperp[:,1] > self.redmin, rperp[:,1] < self.redmax))
        rbins = [np.min(rperp),25,50,75,100,125,150,175,200,225,250,275]
        (hists, rbins) = np.histogram(rperp[ind,0][0], rbins)
        conv = self.hubble/self.atime
        rbins *= conv
        phi = 2*math.pi*np.random.random_sample(num)
        rr = np.empty_like(phi)
        total = 0
        for ii in range(np.size(hists)-1):
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
        cofm=np.array([rr*np.cos(phi), rr*np.sin(phi)]).T
        assert np.shape(cofm) == (np.size(rr),2)
        return cofm

    def save_file(self):
        """Save a file including last time in star"""
        return laststar.LastStar.save_file(self)

    def load_savefile(self, savefile=None):
        laststar.LastStar.load_savefile(self,savefile)


def do_stuff(snap, base):
    """Make lines"""
    reds = {5:(1.,2.25), 4:(2.25, 2.75), 3:(2.75,5)}
    halo = DLANrSpectra(snap,base,4000, reds[snap][0], reds[snap][1])
    halo.get_tau("C",4,1548, force_recompute=False)
    halo.get_tau("H",1,1215, force_recompute=False)
    halo.get_tau("C",2,1334, force_recompute=False)
    halo.get_tau("Si",2,1526, force_recompute=False)
    halo.get_tau("Si",4,1393, force_recompute=False)
    print("Got taus")
    for ion in (2,3,4,5): #xrange(1,8):
        halo.get_density("C",ion)
    halo.get_density("C",-1)
    halo.get_density("H",1)
    print("Got density")
    halo.save_file()

if __name__ == "__main__":
#     simbase = myname.get_name(5, box=10)
#     do_stuff(5, simbase)
#     simbase = myname.get_name(7, box=7.5)
#     do_stuff(5, simbase)
    for ss in (4, 7, 9):
        simbase = myname.get_name(ss, box=25)
        do_stuff(3, simbase)
        do_stuff(4, simbase)
        do_stuff(5, simbase)
