"""Compute some quantities requested by Kathy Cooksey at z=2
as a training set for her student project.
These spectra are CIV-centric"""
import os.path as path
import numpy as np
import fake_spectra.randspectra as rs
import fake_spectra.spec_utils as ss
import fake_spectra.halocat as halocat
import civ_plotspectra as cp
import myname

class SomeSpectra(cp.CIVPlottingSpectra, rs.RandSpectra):
    """Class which gets __init__ from RandSpectra and everything
    else from CIVPlottingSpectra (especially find_nearest_halo)"""
    def __init__(self,num, base, ndla = 10000, numlos=50000, res = 1., thresh=10**20.3, savefile="rand_spectra_DLA.hdf5", savedir=None, elem="H", ion=1):
        rs.RandSpectra.__init__(self,num, base, ndla=ndla, numlos=numlos, res=res, thresh=thresh, savefile=savefile, savedir=savedir, elem=elem, ion=ion)
        (_, self.sub_mass, self.sub_cofm, self.sub_radii) = halocat.find_wanted_halos(num, base, 10)
        assert np.size(self.sub_mass) > 0
        self.savedir=savedir

    def get_cofm(self, num = None):
        """Override to RandSpectra version"""
        return rs.RandSpectra.get_cofm(self, num)

    def save_eqw_txt_data(self, elem, ion, line):
        """Save the equivalent width to a text file"""
        eqw = self.equivalent_width(elem, ion, line)
        np.savetxt(path.join(self.savedir, "eqw_"+str(elem)+str(ion)+"_"+str(line)+".txt"), eqw)

    def vel_width(self, elem, ion, line):
        """
           Find the velocity width of an ion.
           This is the width in velocity space containing 90% of the optical depth
           over the absorber.

           elem - element to look at
           ion - ionisation state of this element.
        """
        try:
            return self.vel_widths[(elem, ion)]
        except KeyError:
            tau = self.get_tau(elem, ion, line)
            (low, high, offset) = self.find_absorber_width(tau)
            #  Size of a single velocity bin
            vel_width = np.zeros(np.shape(tau)[0])
            #deal with periodicity by making sure the deepest point is in the middle
            for ll in np.arange(0, np.shape(tau)[0]):
                tau_l = np.roll(tau[ll,:],offset[ll])[low[ll]:high[ll]]
                (nnlow, nnhigh) = self._vel_width_bound(tau_l)
                vel_width[ll] = self.dvbin*(nnhigh-nnlow)
            #Return the width
            self.vel_widths[(elem, ion)] = vel_width
            return self.vel_widths[(elem, ion)]

    def find_absorber_width(self, strong, chunk = 20, minwidth=None):
        """
           Find the region in velocity space considered to be an absorber for each spectrum.
           This is defined to be the maximum of 1000 km/s and the region over which there is "significant"
           absorption in the strongest line for this ion, where strongest is the line with the largest
           cross-section, ie, greatest lambda * fosc.
           elem, ion - ion to look at

           This line will be highly saturated, so consider significant absorption as F < 3/snr,
           or F < 0.15 for no noise (and an assumed SNR of 20).

           Returns the low and high indices of absorption, and the offset for the maximal absorption.
        """
        if minwidth is None:
            minwidth = self.minwidth
        if self.snr > 0:
            thresh = - np.log(1-4./self.snr)
        else:
            thresh = -np.log(1-0.15)
        (offset, roll) = ss.get_rolled_spectra(strong)
        #Minimum
        if minwidth > 0 and minwidth < self.nbins/2:
            low  = int(self.nbins/2-minwidth/self.dvbin)*np.ones(self.NumLos, dtype=np.int)
            high = int(self.nbins/2+minwidth/self.dvbin)*np.ones(self.NumLos, dtype=np.int)
        else:
            low = np.zeros(self.NumLos, dtype=np.int)
            high = self.nbins*np.ones(self.NumLos, dtype=np.int)
        for ii in range(self.NumLos):
            #First expand the search area in case there is absorption at the edges.
            for i in range(low[ii],0,-chunk):
                if not np.any(roll[ii,i:(i+chunk)] > thresh):
                    low[ii] = i
                    break
            #Where is there no absorption rightwards of the peak?
            for i in range(high[ii],self.nbins,chunk):
                if not np.any(roll[ii,i:(i+chunk)] > thresh):
                    high[ii] = i+chunk
                    break
            #Shrink to width which has some absorption
            ind = np.where(roll[ii][low[ii]:high[ii]] > thresh)[0]
            if np.size(ind) != 0:
                oldlow = low[ii]
                low[ii] = np.max((ind[0]+oldlow,0))
                high[ii] = np.min((ind[-1]+oldlow+chunk,self.nbins))
        return (low, high, offset)


    def _vel_width_bound(self, tau):
        """Find the 0.05 and 0.95 bounds of the integrated optical depth"""
        #Zero everything less than 1 sigma significant
        cum_tau = np.cumsum(tau)
        #Use spline interpolation to find the edge of the bins.
        tdiff = cum_tau - 0.95*cum_tau[-1]
        high = np.where(tdiff >= 0)[0][0]
        tdiff = cum_tau - 0.05*cum_tau[-1]
        low = np.where(tdiff >= 0)[0][0]
        return (low, high)

    def save_vw_txt_data(self, elem, ion, line):
        """Save the equivalent width to a text file"""
        v90 = self.vel_width(elem, ion, line)
        np.savetxt(path.join(self.savedir, "v90_"+str(elem)+str(ion)+"_"+str(line)+".txt"), v90)

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
