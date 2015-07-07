# -*- coding: utf-8 -*-
"""This module contains a special class for computing ."""
from __future__ import print_function
import spectra as sp
import os.path as path
import numpy as np
import h5py
from check_photocollis import CloudyPhotoTable

class CollisionalSpectra(sp.Spectra):
    """Override get_elem_den to get the density of element which is collisionally ionised"""
    def _get_elem_den(self, elem, ion, den, temp, data, ind, ind2, star):
        """Get the density in a collisionally ionised elemental species."""
        #First get the actual elemental density using the parent class.
        elem_den = sp.Spectra._get_elem_den(self, elem, ion, den, temp, data, ind, ind2, star)
        #Next get whether we are collisionally ionised.
        #Load the table of collisional vs photoionisation rates into a CloudyPhotoTable
        try:
            self.cloudy_table_collis.ratio(1,1e5)
        except AttributeError:
            #Note that if elem != C or ion !=4 this will fail as the table has not been computed.
            self.cloudy_table_collis = CloudyPhotoTable(self.red, path.expanduser("~/codes/cloudy_tables/get_rate"), elem, ion)
        #Ditto density: high densities are not followed correctly
        denslimits = self.cloudy_table_collis.get_dens_bounds()
        if np.max(den) > denslimits[1]:
            den2 = np.array(den)
            den2[np.where(den2 > denslimits[1])] = denslimits[1]
        else:
            den2 = den
        #All particles with collisional fraction < 0.5 get ignored
        elem_den = elem_den * np.float32(self.cloudy_table_collis.ratio(den2, temp) > 0.5)
        return elem_den

    def save_file(self):
        """
        Saves spectra to a file, because they are slow to generate.
        File is by default to be $snap_dir/snapdir_$snapnum/spectra.hdf5.
        """
        #We should make sure we have loaded all lazy-loaded things first.
        try:
            f=h5py.File(self.savefile,'r+')
        except IOError:
            raise IOError("Could not open ",self.savefile," for writing")
        grp_grid = f.create_group("colden_collis")
        self._save_multihash(self.colden, grp_grid)
        grp_grid = f.create_group("tau_collis")
        self._save_multihash(self.tau, grp_grid)

if __name__ == "__main__":
    cs = CollisionalSpectra(5,"/home/spb/data/Cosmo/Cosmo7_V6/L25n512/output/",None, None, savefile="rand_civ_spectra.hdf5")
    c4collis = cs.get_col_density("C",4,force_recompute=True)
    cs.save_file()
