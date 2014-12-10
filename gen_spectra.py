"""Short module to generate randomly positioned CIV spectra for computation of Omega_CIV"""
import randspectra as rs

def get_civ(sim, box):
    """Generate the spectra through random directions"""
    base = myname.get_name(sim,box=box)
    #Resolution: Cooksey is SDSS which is ~100 km/s, but D'Odorico is HIRES.
    for n in xrange(6):
        ahalo = rs.RandSpectra(n, base, numlos=5000, thresh=0., res=2., savefile="rand_civ_spectra.hdf5")
        ahalo.get_tau("C",4,1548, force_recompute=True)
        ahalo.get_tau("C",-1,1548, force_recompute=True)
        ahalo.get_density("C",4)
        ahalo.get_density("C",2)
        ahalo.get_density("C",-1)
        ahalo.get_velocity("C",4)
        ahalo.get_density("H",1)
        ahalo.save_file()

for ss in (0,1,3,4,5,7,9):
    get_civ(ss, 25)

get_civ(5, 10)
get_civ(7, 7.5)

