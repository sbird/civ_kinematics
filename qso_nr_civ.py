# -*- coding: utf-8 -*-
"""Class to generate CIV spectra that are nearby to a DLA"""

import numpy as np
import math
import spectra
import myname
import halocat
import hdfsim

class QSONrSpectra(spectra.Spectra):
    """Generate metal line spectra from simulation snapshot"""
    def __init__(self,num, base, numlos, min_mass, res = 5., cdir = None, savefile="nr_qso_spectra.hdf5", savedir=None, reload_file=True):
        #Load halos to push lines through them
        f = hdfsim.get_file(num, base, 0)
        self.OmegaM = f["Header"].attrs["Omega0"]
        self.box = f["Header"].attrs["BoxSize"]
        self.hubble = f["Header"].attrs["HubbleParam"]
        self.atime = f["Header"].attrs["Time"]

        self.npart=f["Header"].attrs["NumPart_Total"]+2**32*f["Header"].attrs["NumPart_Total_HighWord"]
        f.close()
        (_, self.sub_mass, cofm, self.sub_radii) = halocat.find_wanted_halos(num, base, min_mass)
        self.sub_cofm = np.array(cofm, dtype=np.float64)
        self.NumLos = numlos
        #All through y axis
        axis = np.ones(self.NumLos)
        #Re-seed for repeatability
        np.random.seed(23)
        select = np.random.randint(0,np.size(self.sub_mass),numlos)
        #Select the spectra
        cofm = np.concatenate((self.sub_cofm[select], self.sub_cofm[select]))
        axis = np.concatenate((axis[select], axis[select]))
        #Add a small perturbation to the sightline cofm
        axx = set([0,1,2])
        rands = self.get_weighted_perp(numlos)
        for i in np.arange(0,numlos):
            ax = axx - set([axis[numlos+i]])
            cofm[numlos+i, list(ax)] += rands[i]
        spectra.Spectra.__init__(self,num, base, cofm, axis, res, cdir, savefile=savefile,savedir=savedir,reload_file=reload_file)

    def get_cofm(self, num = None):
        """Find a bunch more sightlines: do nothing in this case"""
        return num

    def get_rand_pert(self, num, minradius, maxradius):
        """Get random perturbations within a projected distance of radius specified in proper kpc"""
        #Convert from proper kpc to comoving kpc/h
        conv = self.hubble/self.atime
        minradius *= conv
        maxradius *= conv
        #Generate random sphericals
        #theta = 2*math.pi*np.random.random_sample(num)-math.pi
        phi = 2*math.pi*np.random.random_sample(num)
        rr = (maxradius-minradius)*np.random.random_sample(num) + minradius
        #Add them to halo centers
        cofm = np.empty((num, 2), dtype=np.float64)
        cofm[:,0]=rr*np.cos(phi)
        cofm[:,1]=rr*np.sin(phi)
        return cofm

    def get_weighted_perp(self, num):
        """Get a random perturbation with a radius weighted by the number of
           quasars in radial bins in the QPQ survey.
           This needs updating with better lines"""
        rperp = np.loadtxt("apj503418t2_mrt.txt")
        rbins = np.arange(np.min(rperp), 1000, 100)
        (hists, rbins) = np.histogram(rperp, rbins)
        conv = self.hubble/self.atime
        rbins *= conv
        phi = 2*math.pi*np.random.random_sample(num)
        rr = np.empty_like(phi)
        total = 0
        for ii in xrange(np.size(hists)):
            #How many sightlines in this bin?
            #The proportion from the observed survey, but at least 1 and no more
            #than we have left in the bag
            this = np.min((np.max((int(num*hists[ii]/1./np.sum(hists)),1)), num - total))
            rr[total:total+this] = (rbins[ii+1] - rbins[ii])*np.random.random_sample(this) + rbins[ii]
            total+=this
        cofm = np.empty((num, 2), dtype=np.float64)
        cofm[:,0]=rr*np.cos(phi)
        cofm[:,1]=rr*np.sin(phi)
        return cofm

def do_stuff(snap, path):
    """Make lines"""
    halo = QSONrSpectra(snap,path,1000, 12.5)
    halo.get_tau("C",4,1548, force_recompute=False)
    halo.get_tau("H",1,1215, force_recompute=False)
    halo.get_tau("C",-1,1548, force_recompute=False)
    halo.get_tau("C",2,1334, force_recompute=False)
    halo.get_tau("Si",2,1526, force_recompute=False)
    halo.get_tau("Si",4,1393, force_recompute=False)
    for ion in xrange(1,8):
        halo.get_density("C",ion)
    halo.get_density("C",-1)
    halo.get_velocity("C",4)
    halo.get_density("H",1)
    halo.save_file()

if __name__ == "__main__":
#    simbase = myname.get_name(5, box=10)
#    do_stuff(5, simbase)
#    simbase = myname.get_name(7, box=7.5)
#    do_stuff(5, simbase)
   for ss in (1, 3, 4, 5, 6, 7, 9):
       simbase = myname.get_name(ss, box=25)
       do_stuff(5, simbase)
    #for ss in (4,7):
        #simbase = myname.get_name(ss, box=25)
        #do_stuff(3, simbase)
        #do_stuff(1, simbase)
