"""Make some plots to check the effects of various things."""
from __future__ import print_function
import os.path as path
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.integrate
import myname
import randspectra as rs
import civ_data
import subfindhdf
from civ_plotspectra import CIVPlottingSpectra
from save_figure import save_figure
try:
    xrange(1)
except NameError:
    xrange = range

outdir = path.join(myname.base, "civ_plots/")

print("Plots at ",outdir)

#colors = {0:"pink", 1:"purple", 2:"cyan", 3:"green", 4:"gold", 5:"red", 7:"blue", 6:"grey", 8:"pink", 9:"orange", 'A':"grey", 'S':"red"}
lss = {0:"--",1:":", 2:":",3:"-.", 4:"-.", 5:"-",6:"--",7:":", 8:"-",9:"--",'A':"--",  'S':"--",'VS':"-", 'I':"-"}
#labels = {0:"ILLUS",1:"HVEL", 2:"HVNOAGN",3:"NOSN", 4:"WMNOAGN", 5:"MVEL",6:"METAL",7:"DEF", 8:"RICH",9:"FAST", 'A':"MOM", 'S':"SMALL"}
labels = {1:"HVEL", 2:"HVNOAGN", 3:"NOSN", 5:"MVEL", 7:"ILLUS", 9:"FAST", 4:"WARM",'S':"SMALL", 'VS':"VSMALL",6:"LOAD", 'I':"ILLUS"}
colors = {1:"purple", 2:"grey", 3:"green", 5:"yellow", 7:"#ca0020", 9:"#92c5de", 4:"#0571b0",6:"green",'I':"#f4a582",'S':"grey", 'VS':"brown"}
#colors = {1:"purple", 2:"grey", 3:"green", 5:"yellow", 7:"#348ABD", 9:"#7A68A6", 4:"#467821",'S':"grey", 'VS':"brown",6:"green", 'I':"#CF4457"}
#colors = {1:"purple", 2:"grey", 3:"green", 5:"yellow", 7:"blue", 9:"magenta", 4:"green",'S':"grey", 'VS':"brown",6:"green", 'I':"brown"}
snaps = {1:4, 2:3.5, 3:3, 4:2.5, 5:2, 6:1.5, 7:1, 8:0.5, 9:0.3, 10:0, 901:6, 902:5, 903:4.5, 54:4.0, 57:3.5, 60:3.0, 64:2.5, 68:2.0}

def plot_voigt_cddf(sim, box=25, snap=5):
    """Plot the cddf as compared to D'Odorico 2010, accounting for redshift distribution"""
    base = myname.get_name(sim, box=box)
    try:
        ahalo = CIVPlottingSpectra(snap, base, None, None, savefile="rand_civ_spectra_small.hdf5", spec_res=5.,load_halo=False)
    except IOError:
        ahalo = rs.RandSpectra(snap, base, ndla=1500, numlos=6000, thresh=1e11, res=5., savefile="rand_civ_spectra_small.hdf5", elem="C", ion=4)
        ahalo.get_tau("C",4,1548)
        ahalo.save_file()
    (NHI, cddf) = ahalo.column_density_function("C", 4, minN=11.5,maxN=16.5, line=False, close=50.)
    plt.loglog(NHI,cddf,color="blue", label="Summation", ls="-")
    (NHI_voigt, cddf_voigt) = ahalo.column_density_from_voigt("C", 4, line=1548, minN=11.5,maxN=16.5, close=50.)
    plt.loglog(NHI_voigt,cddf_voigt,color="red", label="Voigt", ls="--")
    ax=plt.gca()
    ax.set_xlabel(r"$N_\mathrm{CIV} (\mathrm{cm}^{-2})$")
    ax.set_ylabel(r"$f(N) (\mathrm{cm}^2)$")
    plt.xlim(10**12, 10**15)
    plt.legend(loc=0)

def plot_resolution_cddf(snap=3, maxfac=1.):
    """Plot the effect of changing resolution on the CDDF."""
    base_large = myname.get_name(7, box=25)
    base_small = myname.get_name(7, box=7.5)
    ahalo_large = CIVPlottingSpectra(snap, base_large, None, None, savefile="rand_civ_spectra.hdf5", spec_res=5.,load_halo=True)
    ahalo_small = CIVPlottingSpectra(snap, base_small, None, None, savefile="rand_civ_spectra.hdf5", spec_res=5.,load_halo=True)
    maxmass = np.max(ahalo_small.sub_mass)/maxfac
    print("Max mass=",maxmass/1e10," was ",np.max(ahalo_large.sub_mass)/1e10)
    print("Small box has ",np.size(np.where(ahalo_small.sub_mass > maxmass))," larger halos")
    print("Larger box has ",np.size(np.where(ahalo_large.sub_mass > maxmass))," larger halos")
    ahalo_large.get_col_density("C",4)
    ahalo_small.get_col_density("C",4)
    (halos_large,_) = ahalo_large.find_nearest_halo("C",4, thresh=50)
    (halos_small,_) = ahalo_small.find_nearest_halo("C",4, thresh=50)
    ind_large = np.where((ahalo_large.sub_mass[halos_large] < maxmass)*(halos_large > 0))
    ind_small = np.where((ahalo_small.sub_mass[halos_small] < maxmass)*(halos_small > 0))
    print("Now ",np.size(ind_large),np.size(ind_small)," spectra")
    #Editing the private data like this is perilous
    ahalo_large.colden[("C",4)] = ahalo_large.colden[("C",4)][ind_large]
    ahalo_small.colden[("C",4)] = ahalo_small.colden[("C",4)][ind_small]
    (NHI_large, cddf_large) = ahalo_large.column_density_function("C", 4, minN=11.5,maxN=16.5, line=False, close=50.)
    plt.loglog(NHI_large,cddf_large,color="blue", label="25 Mpc Box", ls="-")
    (NHI_small, cddf_small) = ahalo_small.column_density_function("C", 4, minN=11.5,maxN=16.5, line=False, close=50.)
    plt.loglog(NHI_small,cddf_small,color="red", label="7.5 Mpc Box", ls="--")
    ax=plt.gca()
    ax.set_xlabel(r"$N_\mathrm{CIV} (\mathrm{cm}^{-2})$")
    ax.set_ylabel(r"$f(N) (\mathrm{cm}^2)$")
    plt.xlim(10**12, 10**15)
    plt.legend(loc=0)
    ax=plt.gca()
    ax.set_xlabel(r"$N_\mathrm{CIV} (\mathrm{cm}^{-2})$")
    ax.set_ylabel(r"$f(N) (\mathrm{cm}^2)$")

def plot_UVB_cddf(box=25, snap=5):
    """Plot the cddf as compared to D'Odorico 2010, accounting for redshift distribution"""
    base = myname.get_name(7, box=box)
    try:
        ahalo_double = CIVPlottingSpectra(snap, base, None, None, savefile="rand_civ_spectra_double.hdf5", spec_res=5.,load_halo=False, cdir=path.expanduser("~/codes/cloudy_tables/ion_out_double"))
    except IOError:
        ahalo_double = rs.RandSpectra(snap, base, ndla=5000, numlos=10000, thresh=1e12, res=5., savefile="rand_civ_spectra_double.hdf5", elem="C", ion=4)
        ahalo_double.get_tau("C",4,1548)
        ahalo_double.save_file()
    base = myname.get_name(0, box=box)
    ahalo = CIVPlottingSpectra(snap, base, None, None, savefile="rand_civ_spectra.hdf5", spec_res=5.,load_halo=False)
    (NHI, cddf) = ahalo.column_density_function("C", 4, minN=11.5,maxN=16.5, line=False, close=50.)
    plt.loglog(NHI,cddf,color="blue", label="FG09 UVB", ls="-")
    (NHI_double, cddf_double) = ahalo_double.column_density_function("C", 4, minN=11.5,maxN=16.5, line=False, close=50.)
    plt.loglog(NHI_double,cddf_double,color="red", label="2x FG09 UVB", ls="--")
    ax=plt.gca()
    ax.set_xlabel(r"$N_\mathrm{CIV} (\mathrm{cm}^{-2})$")
    ax.set_ylabel(r"$f(N) (\mathrm{cm}^2)$")
    plt.xlim(10**12, 10**15)
    plt.legend(loc=0)
    ax=plt.gca()
    ax.set_xlabel(r"$N_\mathrm{CIV} (\mathrm{cm}^{-2})$")
    ax.set_ylabel(r"$f(N) (\mathrm{cm}^2)$")

def abs_dist_int(zz):
    """Absorption distance integration kernel."""
    Hz = np.sqrt(0.3*(1+zz)**3 + 0.7)
    return (1+zz)**2/Hz

def absorption_distance(red1, red2):
    """
    Compute X(z), the absorption distance per sightline (dimensionless)
    X(z) = int (1+z)^2 H_0 / H(z) dz
    """
    return scipy.integrate.quad(abs_dist_int, red1, red2)[0]

def load_rahmati():
    """Load the CDDF from Ali Rahmati's txtfile."""
    eagle = np.loadtxt("others/EAGLEL100N1504_s16_z2p0_CIV_10kpix_axis0.txt")
    return 10**eagle[:,0], 10**eagle[:,1]

def load_oppenheimer():
    """Load the CDDF from Ben Oppenheimer's txtfile."""
    opp = np.loadtxt("others/cdd.r48n384vzw15p.c4.z15_20.hist")
    lnhilow = opp[:,0]+13
    lnhihigh = opp[:,1]+13.
    lnhi = (lnhihigh + lnhilow)/2.
    nhi = 10**lnhi
    #The sixth column contains the number of absorbers along 70 sightlines.
    nabs = opp[:,5]/70.
    #Compute dN
    dN = 10**lnhihigh - 10**lnhilow
    #Difference in absorption distance between z=1.5 and z=2.
    dX = absorption_distance(1.5,2.0)
    f_N = nabs/dN/dX
    return nhi, f_N

def plot_compare_others_cddf(sim=4):
    """Compare our results for the CIV cddf to those of other people, Ali Rahmati and Ben Oppenheimer.
    Because they just do it at z=2, use only one snapshot."""
    base = myname.get_name(sim, box=25)
    (NHI_opp, cddf_opp) = load_oppenheimer()
    plt.loglog(NHI_opp,cddf_opp,color="grey", label="Oppenheimer+2012", ls="-")
    ahalo = CIVPlottingSpectra(5, base, None, None, savefile="rand_civ_spectra.hdf5", spec_res=5.,load_halo=False)
    (NHI, cddf) = ahalo.column_density_function("C", 4, minN=11.5,maxN=16.5, line=False, close=50.)
    plt.loglog(NHI,cddf,color=colors[4], label=labels[4], ls=lss[4])
    (NHI_rah, cddf_rah) = load_rahmati()
    plt.loglog(NHI_rah,cddf_rah,color="brown", label="EAGLE", ls="--")
    civ_data.plot_dor_cddf()
    plt.legend(loc=0)
    plt.xlim(1e12,5e15)
    ax=plt.gca()
    ax.set_xlabel(r"$N_\mathrm{CIV} (\mathrm{cm}^{-2})$")
    ax.set_ylabel(r"$f(N) (\mathrm{cm}^2)$")

def plot_some_spectra(sim=4):
    """Plot a few CIV spectra."""
    base = myname.get_name(sim, box=25)
    ahalo = CIVPlottingSpectra(5, base, None, None, savefile="rand_civ_spectra.hdf5", spec_res=5.,load_halo=False)
    maxt = np.max(ahalo.get_tau("C",4,1548), axis=1)
    large = np.where((maxt > 4)*(maxt < 10))[0]
    xlim=150
    for ll in large[:30]:
        ahalo.plot_spectrum("C",4,1548, ll, flux=True, xlims=(-500-3*xlim,xlim), color="black")
        ahalo.plot_spectrum("C",4,1550, ll, flux=True, xlims=(-500-3*xlim,xlim), color="grey",ls="-",offset=-500)
        save_figure(path.join(outdir,"spectra/civ_spectrum_"+str(ll)))
        plt.clf()
    small = np.where((maxt < 1)*(maxt > 0.2))[0]
    for ll in small[:30]:
        ahalo.plot_spectrum("C",4,1548, ll, flux=True, xlims=(-500-3*xlim,xlim), color="black")
        ahalo.plot_spectrum("C",4,1550, ll, flux=True, xlims=(-500-3*xlim,xlim), color="grey",ls="-",offset=-500)
        save_figure(path.join(outdir,"spectra/civ_spectrum_"+str(ll)))
        plt.clf()

def plot_stellar_mass_function(sim, snap=5):
    """Plot the galaxy stellar mass function for a snapshot."""
    base = myname.get_name(sim, box=25)
    subs=subfindhdf.SubFindHDF5(base, snap)
    stellar_mass = subs.get_grp("GroupMassType")[:,4]*1e10/0.7
    #Could also use subhalo stellar mass: they are similar
    #stellar_mass = subs.get_sub("SubhaloMassType")[:,4]
    bins = np.logspace(6,11)
    dlogM = np.diff(np.log10(bins[:2]))
    volume = (25/0.7)**3
    (gsmf,sm) = np.histogram(stellar_mass, bins=bins)
    sm = (sm[1:]+sm[:-1])/2.
    assert np.size(sm) == np.size(gsmf)
    gsmf = gsmf/volume/dlogM
    plt.loglog(sm, gsmf, color=colors[sim], ls=lss[sim], label=labels[sim])

def plot_galactic_metallicity(sim, snap=5):
    """Plot histograms of the stellar metallicity."""
    base = myname.get_name(sim, box=25)
    subs=subfindhdf.SubFindHDF5(base, snap)
    stellar_metallicity = subs.get_grp("GroupStarMetallicity")
    solarz = 0.0134/0.7381
    bins = np.logspace(-8,0)
    (metallicity,sm) = np.histogram(stellar_metallicity, bins=bins, density=True)
    sm = (sm[1:]+sm[:-1])/2./solarz
    plt.loglog(sm, metallicity, color=colors[sim], ls=lss[sim], label=labels[sim])

def do_stellar_metallicity_plot(snap=5):
    """Plot the galaxy stellar metallicity."""
    for sim in (4,7,9):
        plot_galactic_metallicity(sim, snap)
    plt.legend(loc=0)
    #This is from the MNRAS machine-readable table in http://arxiv.org/abs/1101.2867 z=2-2.5 bin.
    plt.xlim(1e-6,1)
    plt.xlabel(r"$Z / Z_\odot$")
    plt.ylabel(r"Stellar metallicity PDF")

def do_stellar_mass_plot(snap=5):
    """Plot the galaxy stellar mass function."""
    for sim in (4,7,9):
        plot_stellar_mass_function(sim, snap)
    plt.legend(loc=0)
    #This is from the MNRAS machine-readable table in http://arxiv.org/abs/1101.2867 z=2-2.5 bin.
    mo11 = np.loadtxt("mo11_gsmf_z2.txt")
    plt.errorbar(10**mo11[:,0], 10**mo11[:,1], fmt='o',color="black",yerr=[10**(mo11[:,1]+mo11[:,2])-10**mo11[:,1], -10**(mo11[:,1]-mo11[:,2])+10**mo11[:,1]])
    plt.xlim(5e7,1e11)
    plt.xlabel(r"$M_* / M_\odot$")
    plt.ylabel(r"$\Phi$ (Mpc$^{-3}$ dex$^{-1}$)")

if __name__ == "__main__":
    plot_some_spectra(sim=4)
    do_stellar_mass_plot()
    save_figure(path.join(outdir, "Cosmo_stellar_mass_z2"))
    plt.clf()
    do_stellar_metallicity_plot()
    save_figure(path.join(outdir, "Cosmo_stellar_metallicity_z2"))
    plt.clf()
    plot_resolution_cddf()
    plt.ylim(1e-17, 2e-12)
    save_figure(path.join(outdir,"civ_Cosmo7_resolution"))
    plt.clf()
    plot_UVB_cddf()
    plt.ylim(5e-17, 2e-12)
    save_figure(path.join(outdir,"civ_UVB_double"))
    plt.clf()
    plot_compare_others_cddf()
    plt.ylim(5e-17, 2e-12)
    plt.xlim(1e12, 1e15)
    save_figure(path.join(outdir,"civ_compare"))
    plt.clf()
    plot_voigt_cddf(4,25,5)
    save_figure(path.join(outdir,"civ_voigt_Cosmo4"))
    plt.clf()
