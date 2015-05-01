#!/usr/bin env python
# -*- coding: utf-8 -*-
"""Make some plots of the velocity widths from the cosmo runs"""

import matplotlib
matplotlib.use('PDF')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import civ_plots as ps
import os.path as path
import myname
import numpy as np
from save_figure import save_figure

np.seterr(divide='raise',invalid='raise')
labels = {1:"HVEL", 3:"NOSN", 5:"MVEL", 7:"ILLUS 25", 9:"FAST 25", 4:"WARM 25",6:"LOAD", 'I':"ILLUS 75"}
colors = {1:"purple", 3:"green", 5:"yellow", 7:"blue", 9:"red", 4:"gold",6:"green",'I':"brown"}
lss = {0:"--",1:":", 2:":",3:"-.", 4:"--", 5:"-",6:"--",7:":", 8:"-",9:"-.",'A':"--",  'S':"--",'VS':"-", 'I':"-"}

outdir = path.join(myname.base, "civ_plots/")

print "Plots at ",outdir

def plot_den(hspec, ax, num, voff = 0, elem="C", ion=4, color="blue"):
    """Plot density"""
    plt.sca(ax[0])
    xoff = hspec.plot_spectrum(elem,ion,1548,num, flux=True, color=color)
    xlim = plt.xlim()
    ax[0].xaxis.set_label_position('top')
    ax[0].xaxis.tick_top()
    voff += xoff
    plt.sca(ax[1])
    dxlim = hspec.plot_density(elem,ion, num, color=color)
    plt.ylabel(r"n$_\mathrm{"+elem+"IV}$ (cm$^{-3}$)")
    #plt.ylim(ymin=1e-9)
    plt.sca(ax[2])
    xscale = dxlim*hspec.velfac/xlim[1]
    hspec.plot_den_to_tau(elem, ion, num, thresh = 1e-9, xlim=200,voff=voff, xscale=xscale)
    ax[2].axes.get_xaxis().set_visible(False)
    plt.xlabel("")
    plt.xlim(xlim)

def do_civ_plots(name, ahalos):
    """Make a bunch of plots"""
    CIV_eq_ratio(name, ahalos)
    generic_coverfrac("CIV", "C", 4, 1548, name, ahalos)
    generic_coverfrac("SiII", "Si", 2, 1526, name, ahalos)
    HI_coverfrac(name, ahalos)
    generic_eq_width("CII", "C", 2, 1334, name, ahalos)
    generic_eq_width("CIV", "C", 4, 1548, name, ahalos)
    generic_eq_width("SiII", "Si", 2, 1526, name, ahalos)
    generic_eq_width("SiIV", "Si", 4, 1393, name, ahalos)
    #generic_eq_width("HI", "H", 1, 1216, name, ahalos)

def CIV_eq_ratio(name, ahalos):
    """Carbon IV equivalent width ratio"""
    for ahalo in ahalos:
        ahalo.plot_eq_width_ratio()
    plt.xlabel("r perp (kpc)")
    plt.ylabel(r"$\eta_\mathrm{pair} / \eta_\mathrm{DLA}$")
    plt.ylim(0, 1.5)
    plt.legend()
    save_figure(path.join(outdir,name+"_CIV_eq_ratio"))
    plt.clf()

def generic_eq_width(ionname, elem, ion, line, name, ahalos):
    """Plot eq. width distribution"""
    for ahalo in ahalos:
        ahalo.plot_eq_width(elem=elem, ion=ion, line=line)
    plt.xlabel("r perp (kpc)")
    plt.ylabel(r"EW("+ionname+" "+str(line)+")")
    CGM_w = np.loadtxt("CGMofDLAs_avgW"+ionname+".dat")
    if np.size(CGM_w[:,0]) == 4:
        plt.errorbar(CGM_w[:,0], CGM_w[:,1], yerr = CGM_w[:,2], xerr=[CGM_w[:,0]-[0,7,100,200],[7,100,200,275]-CGM_w[:,0]], fmt='o',ecolor="black")
    if np.size(CGM_w[:,0]) == 5:
        plt.errorbar(CGM_w[:,0], CGM_w[:,1], yerr = CGM_w[:,2], xerr=[CGM_w[:,0]-[0,7,50,100,200],[7,50,100,200,275]-CGM_w[:,0]], fmt='o',ecolor="black")
    plt.xlim(-5,300)
    plt.legend()
    save_figure(path.join(outdir,name+"_"+ionname+"_eq_width"))
    plt.clf()

def generic_coverfrac(ionname, elem, ion, line, name, ahalos):
    """Plot covering fraction"""
    for ahalo in ahalos:
        ahalo.plot_covering_fraction(elem=elem, ion=ion, line=line)
    plt.xlabel("r perp (kpc)")
    plt.ylabel(r"$F(W_{"+str(line)+r"} > 0.2 \AA)$")
    CGM_c = np.loadtxt("CGMofDLAs_Cf"+ionname+".dat")
    plt.errorbar(CGM_c[:,0], CGM_c[:,2], yerr = [CGM_c[:,2]-CGM_c[:,1],CGM_c[:,3]-CGM_c[:,2]], fmt='o', xerr=[CGM_c[:,0]-[7,100,200],[100,200,275]-CGM_c[:,0]],ecolor="black")
    plt.ylim(0,1.0)
    plt.legend()
    save_figure(path.join(outdir,name+"_"+ionname+"_coverfrac"))
    plt.clf()

def HI_coverfrac(name, ahalos):
    """CIV covering fraction"""
    for ahalo in ahalos:
        ahalo.plot_covering_fraction_colden(elem="H", ion=1)
    plt.xlabel("r perp (kpc)")
    plt.ylabel(r"$F(LLS)$")
    CGM_c = np.loadtxt("CGMofDLAs_Cfothick.dat")
    plt.errorbar(CGM_c[:,0], CGM_c[:,2], yerr = [CGM_c[:,2]-CGM_c[:,1],CGM_c[:,3]-CGM_c[:,2]], fmt='o', xerr=[CGM_c[:,0]-[7,50,117.5,200],[50,117.5,200,275]-CGM_c[:,0]],ecolor="black")
    plt.ylim(0,1.0)
    plt.legend()
    save_figure(path.join(outdir,name+"_LLS_coverfrac"))
    plt.clf()

def vel_offset_rubin():
    """Plot the histogram of the velocity offsets from Rubin 2014.
    These are rather small number statistics."""
    vbins = np.linspace(0,100,6)
    num = np.array([3,3,2,1,2,1])
    plt.bar(vbins, num, width=20, color="black", alpha=0.4, label="Rubin 2014")

def CIV_vel_offset(name, ahalos):
    """Velocity offset of CIV from the DLAs"""
    vel_offset_rubin()
    for ahalo in ahalos:
        ahalo.plot_flux_vel_offset()
    plt.ylabel("N")
    plt.xlabel(r"$v_\mathrm{pair} - v_\mathrm{DLA}$ (km/s)")
    plt.ylim(0,4)
    plt.xlim(0,120)
    plt.yticks(np.arange(0,5),("0","1","2","3","4"))
    plt.legend()
    save_figure(path.join(outdir,name+"_CIV_vel_offset"))
    plt.clf()

def plot_r_offsets(ahalo):
    """Plot the positions of the sightlines relative to DLAs"""
    r_offsets = ahalo.get_offsets()
    rbins = np.linspace(np.min(r_offsets), np.max(r_offsets), 40)
    (hist, edges) = np.histogram(r_offsets,rbins)
    plt.plot((edges[1:]+edges[:-1])/2, hist, ls='-', label=ahalo.label)

def C_ionic_coverfrac(name, ahalo):
    """Plot covering fraction"""
    ahalo.plot_covering_fraction(elem="C", ion=4, line=1548,color="grey",label="C-IV")
    ahalo.plot_covering_fraction(elem="C", ion=-1, line=1548, color="pink", label="C-ALL")
    plt.xlabel("r perp (kpc)")
    plt.ylabel(r"$F(W_{1548} > 0.2 \AA)$")
    CGM_c = np.loadtxt("CGMofDLAs_CfCIV.dat")
    plt.errorbar(CGM_c[:,0], CGM_c[:,2], yerr = [CGM_c[:,2]-CGM_c[:,1],CGM_c[:,3]-CGM_c[:,2]], fmt='o', xerr=[CGM_c[:,0]-[7,100,200],[100,200,275]-CGM_c[:,0]],ecolor="black")
    plt.ylim(0,1.0)
    plt.legend()
    save_figure(path.join(outdir,name+"_CIV_coverfrac"))
    plt.clf()

def C_ionic_eq_width(name, ahalo):
    """Plot eq. width distribution"""
    ahalo.plot_eq_width(elem="C", ion=4, line=1548,color="grey",label="C-IV")
    ahalo.plot_eq_width(elem="C", ion=-1, line=1548, color="pink", label="C-ALL")
    plt.xlabel("r perp (kpc)")
    plt.ylabel(r"EW(CIV 1548)")
    CGM_w = np.loadtxt("CGMofDLAs_avgWCIV.dat")
    plt.errorbar(CGM_w[:,0], CGM_w[:,1], yerr = CGM_w[:,2], xerr=[CGM_w[:,0]-[0,7,100,200],[7,100,200,275]-CGM_w[:,0]], fmt='o',ecolor="black")
    save_figure(path.join(outdir,name+"_CIV_eq_width"))
    plt.clf()

#Column density plots
def rel_c_colden(ahalo):
    """Column density for different carbon ions"""
    ahalo.plot_colden_ratio(color="grey",elem="C",ion=4,ion2=-1, label="CIV")
    ahalo.plot_colden_ratio(color="pink",elem="C",ion=2, ion2=-1, label="CII")
    ahalo.plot_colden_ratio(color="green",elem="C",ion=3, ion2=-1, label="CIII")
    ahalo.plot_colden_ratio(color="blue",elem="C",ion=5, ion2=-1, label="CV")
    #(center, mean_plot_arr6) = ahalo.plot_colden_ratio(color="yellow",elem="C",ion=6, ion2=-1, label="CVI")
    #(center, mean_plot_arr7) = ahalo.plot_colden_ratio(color="red",elem="C",ion=7, ion2=-1, label="CVII")
    #plt.plot(center, mean_plot_arr2+mean_plot_arr3+mean_plot_arr4+mean_plot_arr5+mean_plot_arr6+mean_plot_arr7, color="black")
    plt.legend(loc='upper center', ncol=2)
    save_figure(path.join(outdir,"ion_C_colden_ratio"))
    plt.clf()

def hc_colden(ahalo, upper=270, name="ion", ions=(2,3,4,5,-1)):
    """Plot the column densities for different CIV ions"""
    if 2 in ions:
        ahalo.plot_colden(color="pink",elem="C",ion=2,label="CII", radial_bins = np.logspace(np.log10(7.5), np.log10(upper), 12))
    if 3 in ions:
        ahalo.plot_colden(color="green",elem="C",ion=3, label="CIII", radial_bins = np.logspace(np.log10(7.5), np.log10(upper), 12))
    if 4 in ions:
        ahalo.plot_colden(color="grey",elem="C",ion=4,label="CIV", radial_bins = np.logspace(np.log10(7.5), np.log10(upper), 12))
    if 5 in ions:
        ahalo.plot_colden(color="blue",elem="C",ion=5, label="CV", radial_bins = np.logspace(np.log10(7.5), np.log10(upper), 12))
    #ahalo.plot_colden(color="black",elem="H",ion=1,label="HI")
    if -1 in ions:
        ahalo.plot_colden(color="brown",elem="C",ion=-1, label="Carbon", radial_bins = np.logspace(np.log10(7.5), np.log10(upper), 12))
    plt.yscale('log')
    plt.ylim(1e10, 1e17)
    plt.legend(loc='upper right', ncol=3)
    save_figure(path.join(outdir,name+"_C_colden"+ahalo.label.replace(" ","_")))
    plt.clf()

def qso_eq_width(ionname, elem, ion, line, name, ahalos):
    """Plot covering fraction for QSO pairs"""
    CGM_c = np.loadtxt("QPQ7eqW.dat")
    for ahalo in ahalos:
        ahalo.def_radial_bins = np.concatenate([CGM_c[:,0], [1000,]])
        ahalo.plot_eq_width(elem=elem, ion=ion, line=line)
    plt.xlabel("r perp (kpc)")
    plt.ylabel(r"EW("+ionname+" "+str(line)+")")
    #Rmin Rmax mpair Rmean W1334   sW1334  Rmean   mpair    W1548   sW1548
    if ion == 4:
        plt.errorbar(CGM_c[:,3], CGM_c[:,8], yerr = CGM_c[:,9], fmt='o', xerr=[CGM_c[:,3]-CGM_c[:,0],CGM_c[:,1]-CGM_c[:,3]],ecolor="black")
    elif ion == 2:
        plt.errorbar(CGM_c[:,3], CGM_c[:,4], yerr = CGM_c[:,5], fmt='o', xerr=[CGM_c[:,3]-CGM_c[:,0],CGM_c[:,1]-CGM_c[:,3]],ecolor="black")
    else:
        raise RuntimeError("No data for ion")
    plt.ylim(0,1.0)
    plt.legend()
    save_figure(path.join(outdir,name+"_"+ionname+"_eq_width"))
    plt.clf()

def qso_colden_coverfrac(elem, ion, name, ahalos):
    """Plot covering fraction for QSOs"""
    CGM_c = np.loadtxt("QPQ6fc.dat")
    for ahalo in ahalos:
        ahalo.def_radial_bins = np.concatenate([CGM_c[:,0], [1000,]])
        ahalo.plot_covering_fraction_colden(cd_thresh=10**17.2,elem=elem, ion=ion)
    plt.xlabel("r perp (kpc)")
    plt.ylabel(r"$F(N_{HI} > 10^{17.2} \mathrm{cm}^{-2})$")
    #Rmin Rmax mpair   fc1216     +1s    -1s
    Rmean = (CGM_c[:,0]+CGM_c[:,1])/2
    plt.errorbar(Rmean, CGM_c[:,3], yerr = [CGM_c[:,4],CGM_c[:,5]], fmt='o', xerr=[Rmean-CGM_c[:,0],CGM_c[:,1]-Rmean],ecolor="black")
    plt.ylim(0,1.0)
    plt.legend()
    save_figure(path.join(outdir,name+"_HI_coverfrac"))
    plt.clf()

def qso_coverfrac(ionname, elem, ion, line, name, ahalos):
    """Plot covering fraction for QSOs"""
    CGM_c = np.loadtxt("QPQ7fc.dat")
    for ahalo in ahalos:
        ahalo.def_radial_bins = np.concatenate([CGM_c[:,0], [1000,]])
        ahalo.plot_covering_fraction(elem=elem, ion=ion, line=line)
    plt.xlabel("r perp (kpc)")
    plt.ylabel(r"$F(W_{"+str(line)+r"} > 0.2 \AA)$")
    #Rmin Rmax mpair   fc1334     +1s    -1s   mpair fc1548  +1s      -1s
    Rmean = (CGM_c[:,0]+CGM_c[:,1])/2
    if ion == 4:
        plt.errorbar(Rmean, CGM_c[:,7], yerr = [CGM_c[:,8],CGM_c[:,9]], fmt='o', xerr=[Rmean-CGM_c[:,0],CGM_c[:,1]-Rmean],ecolor="black")
    elif ion == 2:
        plt.errorbar(Rmean, CGM_c[:,3], yerr = [CGM_c[:,4], CGM_c[:,5]], fmt='o', xerr=[Rmean-CGM_c[:,0],CGM_c[:,1]-Rmean],ecolor="black")
    else:
        raise RuntimeError("No data for ion")
    plt.ylim(0,1.0)
    plt.legend()
    save_figure(path.join(outdir,name+"_"+ionname+"_coverfrac"))
    plt.clf()

def do_qso_plots(name, ahalos):
    """Equivalent width and covering fractions for quasars"""
    qso_coverfrac("CIV", "C", 4, 1548, name, ahalos)
    qso_coverfrac("CII", "C", 2, 1334, name, ahalos)
    qso_eq_width("CII", "C", 2, 1334, name, ahalos)
    qso_eq_width("CIV", "C", 4, 1548, name, ahalos)
    qso_colden_coverfrac("H",1,name, ahalos)

qsos = []
#for i in (7,9):
#   base = myname.get_name(i, box=25)
#   qsos.append(ps.AggCIVPlot((3,4,5), base, redfile = "QSORperpred.txt", savefile="nr_qso_spectra.hdf5", color=colors[i], label=labels[i], spec_res = 10.))

base = path.expanduser("~/data/Illustris/")
illsnaps = (60,63,65,66,68)
qsos.append(ps.AggCIVPlot(illsnaps, base, numlos=35000, redfile = "QSORperpred.txt", savefile="nr_qso_spectra.hdf5", color=colors['I'], label=labels['I'], spec_res = 10.,load_halo=False,velsize=1500))

do_qso_plots("qso", qsos)

hc_colden(qsos[0], upper=1000, name="qso", ions=(2,4))

for k in illsnaps:
    qsos.append(ps.AggCIVPlot(k, base, redfile = "QSORperpred.txt", savefile="nr_qso_spectra.hdf5", color=None, label=labels['I']+" "+str(k), spec_res = 10.,load_halo=False, velsize=1500))

do_qso_plots("small_qso", qsos)

[plot_r_offsets(q) for q in qsos]
plt.legend()
save_figure(path.join(outdir,"QSO_illus_r_offset"))
plt.clf()

print "Done QSO"
aahalos = []

#Plot some properties of the small box only
#name = myname.get_name(7, box=7.5)

#ahalo = ps.CIVPlot(5, name, savefile="nr_dla_spectra.hdf5", label="VSMALL", spec_res = 50.)
#ahalo.color = "brown"

#ahalos = [ahalo,]
for ss in (4,9,7): #Removed 3 and 1 as they don't match DLA properties
    base = myname.get_name(ss, box=25)
    halo = ps.AggCIVPlot((4,5), base, numlos=14000, color=colors[ss], redfile = "DLARperpred.txt", savefile="nr_dla_spectra.hdf5", label=labels[ss], spec_res = 50.,load_halo=False)
    aahalos.append(halo)

#Add Illustris
illhalo = ps.AggCIVPlot((63,68), path.expanduser("~/data/Illustris/"),  numlos=14000, color=colors['I'], redfile = "DLARperpred.txt",savefile="nr_dla_spectra.hdf5", label=labels["I"], spec_res = 50.,load_halo=False)
aahalos.append(illhalo)

CIV_vel_offset("feed", [illhalo,])
do_civ_plots("feed",aahalos)

plot_r_offsets(aahalos[-2])
plt.legend()
save_figure(path.join(outdir,"CIV_r_offset"))
plt.clf()


#C_ionic_coverfrac("ion",aahalos[-2])
#C_ionic_eq_width("ion",aahalos[-2])

#rel_c_colden(ahalo)
hc_colden(aahalos[0])
hc_colden(aahalos[-2])
hc_colden(aahalos[-1])

#Do redshift evolution
# for nn in (7,4):
#     name = myname.get_name(nn, box=25)
#     reds = []
#     zzz = {1:4, 3:3, 5:2}
#     colorz = {1:"blue", 3:"green", 5:"red"}
#     for i in (1,3,5):
#         tmp = ps.CIVPlot(i, name, savefile="nr_dla_spectra.hdf5", label=labels[nn]+" z="+str(zzz[i]), spec_res = 50.)
#         tmp.color=colorz[i]
#         reds.append(tmp)
#     do_civ_plots("redshift_"+str(nn),reds)
#
def print_pretty_spectra(snapnum, simname):
    """Print pretty versions of spectra from a simulation snapshot"""
    rands = np.random.randint(0,1000,20)
    halo = ps.CIVPlot(snapnum, simname, savefile="nr_dla_spectra.hdf5", spec_res = 50.)
    offsets = halo.get_offsets()[rands]
    np.savetxt("tau_Rperp_table.txt", np.sort(np.vstack((np.arange(0,1000), halo.get_offsets())).T,0))
    for nn in rands:
        gs = gridspec.GridSpec(9,2)
        axes = (plt.subplot(gs[0:4,0]), plt.subplot(gs[5:,0]), plt.subplot(gs[4,0]))
        #Adjust the default plot parameters, which do not scale well in a gridspec.
        matplotlib.rc('xtick', labelsize=8)
        matplotlib.rc('ytick', labelsize=8)
        matplotlib.rc('axes', labelsize=8)
        matplotlib.rc('font', size=6)
        matplotlib.rc('lines', linewidth=1.5)
        plot_den(halo, axes, nn+1000, color="red")
        plot_den(halo, axes, nn)
        np.savetxt(str(nn)+"_tau_DLA.txt",halo.get_tau("C",4,1548,nn))
        np.savetxt(str(nn)+"_tau_CGM.txt",halo.get_tau("C",4,1548,nn+1000))
        axes[0].text(-500, 0.2,"offset (prop kpc): "+str(offsets[nn]*0.33333/0.7))
        odir = path.join(outdir, "spectra")
        save_figure(path.join(odir,str(nn)+"_cosmo"+"_CIV_spec"))
        plt.clf()
        matplotlib.rc_file_defaults()


# print_pretty_spectra(5, myname.get_name(7, box=25))

