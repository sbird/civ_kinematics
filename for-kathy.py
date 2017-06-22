"""Compute some quantities requested by Kathy Cooksey at z=2
as a training set for her student project.
These spectra are CIV-centric"""
import fake_spectra.randspectra as rs
import civ_plotspectra as cp
import os.path as path
import numpy as np
import myname

class SomeSpectra(cp.CIVPlottingSpectra, rs.RandSpectra):
    """Class which gets __init__ from RandSpectra and everything
    else from CIVPlottingSpectra (especially find_nearest_halo)"""
    def __init__(self,num, base, ndla = 10000, numlos=50000, res = 1., thresh=10**20.3, savefile="rand_spectra_DLA.hdf5", savedir=None, elem="H", ion=1):
        rs.RandSpectra.__init__(self,num, base, ndla=ndla, numlos=numlos, res=res, thresh=thresh, savefile=savefile, savedir=savedir, elem=elem, ion=ion)
        self.savedir=savedir

    def get_cofm(self, num = None):
        """Override to RandSpectra version"""
        return rs.RandSpectra.get_cofm(self, num)

    def save_eqw_txt_data(self, elem, ion, line):
        """Save the equivalent width to a text file"""
        eqw = self.equivalent_width(elem, ion, line)
        np.savetxt(path.join(self.savedir, "eqw_"+str(elem)+str(ion)+"_"+str(line)+".txt"), eqw)

base = myname.get_name(7,box=25)
ahalo = SomeSpectra(5, base, ndla = 10000, numlos=50000, thresh=1e12, res=5., savefile="rand_kathy_spectra.hdf5",savedir=path.expanduser("~/data/Cosmo/Cosmo7_V6/L25n512/output/for-kathy"), elem="C", ion=4)

#Save the desired equivalent widths
ahalo.save_eqw_txt_data("C",4,1548)
ahalo.save_eqw_txt_data("C",4,1550)
ahalo.save_eqw_txt_data("Si",4,1393)
ahalo.save_eqw_txt_data("Si",4,1402)
ahalo.save_eqw_txt_data("Mg",2,2796)
ahalo.save_eqw_txt_data("Mg",2,2803)

#This is the nearest halo in units of
#virial radius to the mass-weighted CIV spectrum density.
(halos, dists) = ahalo.find_nearest_halo()
#Get halo mass
masses = ahalo.sub_mass[halos]
#Convert distances to physical kpc from virial radii
dists = ahalo.sub_radii[halos]*dists*ahalo.atime/ahalo.hubble
np.savetxt(path.join(ahalo.savedir, "halomass.txt"), masses)
np.savetxt(path.join(ahalo.savedir, "impact.txt"), dists)
