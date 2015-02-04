"""Short module to generate randomly positioned CIV spectra for computation of Omega_CIV"""
import randspectra as rs
import myname

def get_civ(sim, box, start=1, end=10):
    """Generate the spectra through random directions"""
    base = myname.get_name(sim,box=box)
    #Resolution: Cooksey is SDSS which is ~100 km/s, but D'Odorico is HIRES.
    for n in xrange(start,end):
        try:
            ahalo = rs.RandSpectra(n, base, numlos=10000, thresh=1e12, res=5., savefile="rand_civ_spectra.hdf5", elem="C", ion=4)
            ahalo.get_tau("C",4,1548, force_recompute=True)
            ahalo.get_tau("C",-1,1548, force_recompute=True)
            ahalo.get_density("C",4)
            ahalo.get_density("C",-1)
            ahalo.get_density("Mg",2)
            ahalo.get_density("H",1)
            ahalo.save_file()
        except IOError:
            continue

#get_civ(7, 25,1,4)
#This needs to have +"2" added to base above
if __name__ == "__main__":
    for ss in (0,1,2,3,4,5,6,7,9):
        get_civ(ss, 25)

    get_civ(5, 10)
    get_civ(7, 7.5)