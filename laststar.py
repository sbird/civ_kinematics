"""Small module to split out the computation of gas particle 'age'; ie, last time it was in a star"""

import spectra
import numpy as np
import h5py
import hsml
import os.path as path
import shutil

class LastStar(spectra.Spectra):
    """Functions to compute the mass-weighted last time each particle was in a star"""
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
            else:
                return np.zeros([np.shape(self.cofm)[0],self.nbins],dtype=np.float32)

    def get_age(self, elem, ion):
        """Get the column density weighted velocity in each pixel for a given species.
        """
        try:
            self.age
        except AttributeError:
            self.age = {}
        phys = self.dvbin/self.velfac*self.rscale
        try:
            self._really_load_array((elem, ion), self.age, "age")
            return self.age[(elem, ion)]/phys
        except KeyError:
            age =  self._get_mass_weight_quantity(self._age_single_file, elem, ion)
            self.age[(elem, ion)] = age
            return age/phys

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
        except AttributeError:
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
