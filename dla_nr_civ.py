# -*- coding: utf-8 -*-
"""Class to generate CIV spectra that are nearby to a DLA"""

import numpy as np
import math
import spectra

class DLANrSpectra(spectra.Spectra):
    """Generate metal line spectra from simulation snapshot"""
    def __init__(self,num, base, numlos=100, res = 1., cdir = None, savefile="nr_dla_spectra.hdf5", savedir=None):
        #Get a sample of DLAs from the savefile
        dlas = spectra.Spectra(num, base, None, None, res, cdir, savefile="grid_spectra_DLA.hdf5",savedir=savedir)
        dla_cofm = dlas.cofm
        dla_axis = dlas.axis

        self.NumLos = numlos
        #Sightlines at random positions
        #Re-seed for repeatability
        np.random.seed(23)
        select = np.random.randint(0,dlas.NumLos,numlos)
        cofm = np.concatenate((dla_cofm[select], dla_cofm[select]))
        axis = np.concatenate((dla_axis[select], dla_axis[select]))
        #Add a small perturbation to the sightline cofm
        rands = self.get_rand_pert(numlos,25,250)
        cofm[numlos:2*numlos,:] += rands
        spectra.Spectra.__init__(self,num, base, cofm, axis, res, cdir, savefile=savefile,savedir=savedir,reload_file=True)

    def get_cofm(self, num = None):
        """Find a bunch more sightlines: do nothing in this case"""
        return num

    def get_rand_pert(self, num, minradius, maxradius):
        """Get random perturbations within a spherical region"""
        #Generate random sphericals
        theta = 2*math.pi*np.random.random_sample(num)-math.pi
        phi = 2*math.pi*np.random.random_sample(num)
        rr = (maxradius-minradius)*np.random.random_sample(num) + minradius
        #Add them to halo centers
        cofm = np.empty((num, 3), dtype=np.float64)
        cofm[:,0]=rr*np.sin(theta)*np.cos(phi)
        cofm[:,1]=rr*np.sin(theta)*np.sin(phi)
        cofm[:,2]=rr*np.cos(theta)
        return cofm

if __name__ == "__main__":
    base = "/home/spb/data/Cosmo/Cosmo5_V6/L10n512/output/"
    halo = DLANrSpectra(5,base,100)
    halo.get_tau("C",4,1548, force_recompute=True)
    halo.get_density("C",4)
    halo.get_velocity("C",4)
    halo.get_density("H",1)
    halo.save_file()
