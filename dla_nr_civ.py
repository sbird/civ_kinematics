# -*- coding: utf-8 -*-
"""Class to generate CIV spectra that are nearby to a DLA"""

import numpy as np
import math
import spectra
import myname
import hdfsim
import h5py
import hsml
import os.path as path
import shutil

class DLANrSpectra(spectra.Spectra):
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
            cofm[numlos+i, list(ax)] += rands[i]
        spectra.Spectra.__init__(self,num, base, cofm, axis, res, cdir, savefile=savefile,savedir=savedir,reload_file=reload_file)
        self.age = {}

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

    def _age_single_file(self,fn, elem, ion):
        """Get the density weighted interpolated temperature field for a single file"""
        (pos, vel, age, temp, hh, amumass) = self._read_particle_data(fn, elem, ion,True)
        if amumass == False:
            return np.zeros([np.shape(self.cofm)[0],self.nbins],dtype=np.float32)
        else:
            ff = h5py.File(fn, "r")
            data = ff["PartType0"]
            hh2 = hsml.get_smooth_length(data)
            pos2 = np.array(data["Coordinates"],dtype=np.float32)
            ind = self.particles_near_lines(pos2, hh2,self.axis,self.cofm)
            ind2 = self._filter_particles(age, None, None, None)
            part_ids = np.array(data["ParticleIDs"])[ind][ind2]
            assert np.size(part_ids) == np.size(age)
            tracer = ff["PartType3"]
            #Last star time is {a, -a, a+1, a+2} depending on from a star/wind to a gas, or from a gas to a star/wind.
            laststar = np.array(tracer["FluidQuantities"][:,8])
            #Only interested in moving from star or wind to gas.
            #Assume that something from a wind came out of a star reasonably soon before
            #Zero value means was never in a star
            t_ind = np.where(np.logical_and(laststar < 1, laststar != 0))
            laststar = laststar[t_ind]
            #Associate each tracer particle with a gas particle
            tracerparents = np.array(tracer["ParentID"])[t_ind]
            ff.close()
            #Now we have tracers we look through our gas cells to find
            #attached tracers. If we don't find a tracer, set this gas particle to zero time.
            #It was never in a star or wind.
            #Find all particles that don't have attached tracers
            withtracers = np.in1d(part_ids, tracerparents)
            withouttracers = np.where(np.logical_not(withtracers))
            withtracers = np.where(withtracers)
            #Metal density of stuff that has been in a star should be much higher than stuff which hasn't been in a star
            #Otherwise we have problems.
            if np.any(withouttracers):
                print "Metal density in a star ",np.mean(age[withouttracers])
                #It is not allowed to set individual elements of arrays selected in this complex way, so we must create a new array of
                #the multiplication factor.
            if np.size(withtracers) > 0:
                print "metal density not in star ",np.mean(age[withtracers])
                lastfactor = np.array([np.max(np.abs(laststar[np.where(tracerparents == pid)])) for pid in part_ids[withtracers]])
                age = age[withtracers] * lastfactor
                pos = pos[withtracers]
                vel = vel[withtracers]
                temp = temp[withtracers]
                hh = hh[withtracers]
                line = self.lines[("H",1)][1215]
                stuff = self._do_interpolation_work(pos, vel, age, temp, hh, amumass, line, False)
            return stuff

    def get_age(self, elem, ion):
        """Get the column density weighted velocity in each pixel for a given species.
        """
        try:
            self._really_load_array((elem, ion), self.age, "age")
            return self.age[(elem, ion)]
        except KeyError:
            age =  self._get_mass_weight_quantity(self._age_single_file, elem, ion)
            self.age[(elem, ion)] = age
            return age

    def save_file(self):
        """
        Saves spectra to a file, because they are slow to generate.
        File is by default to be $snap_dir/snapdir_$snapnum/spectra.hdf5.
        """
        #We should make sure we have loaded all lazy-loaded things first.
        self._load_all_multihash(self.tau, "tau")
        self._load_all_multihash(self.colden, "colden")
        try:
            self._load_all_multihash(self.velocity, "velocity")
            self._load_all_multihash(self.age, "age")
        except IOError:
            pass
        try:
            if path.exists(self.savefile):
                shutil.move(self.savefile,self.savefile+".backup")
            f=h5py.File(self.savefile,'w')
        except IOError:
            raise IOError("Could not open ",self.savefile," for writing")
        #Temperature
        grp_grid = f.create_group("age")
        self._save_multihash(self.age, grp_grid)
        self._save_file(f)

def do_stuff(snap, base):
    """Make lines"""
    reds = {5:(1.,2.25), 4:(2.25, 5.)}
    halo = DLANrSpectra(snap,base,4000, reds[snap][0], reds[snap][1])
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
#     simbase = myname.get_name(5, box=10)
#     do_stuff(5, simbase)
#     simbase = myname.get_name(7, box=7.5)
#     do_stuff(5, simbase)
    for ss in (4, 7, 9):
        simbase = myname.get_name(ss, box=25)
        do_stuff(5, simbase)
        do_stuff(4, simbase)
