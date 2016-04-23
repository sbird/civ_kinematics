# -*- coding: utf-8 -*-
"""Do some plots for the CIV around quasars in Illustris"""

from __future__ import print_function
import os.path as path
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

import civ_plots as ps
import myname
from save_figure import save_figure
from make_civ_plots import hc_colden,plot_r_offsets

np.seterr(divide='raise',invalid='raise')
labels = {1:"HVEL", 2:"HVNOAGN", 3:"NOSN", 5:"MVEL", 7:"ILLUS 25", 9:"FAST 25", 4:"WARM 25",6:"LOAD", 'I':"ILLUS 75"}
colors = {1:"purple", 2:"grey", 3:"green", 5:"yellow", 7:"#ca0020", 9:"#0571b0", 4:"#92c5de",6:"green",'I':"#f4a582",'S':"grey", 'VS':"brown"}
#colors = {1:"purple", 2:"orange", 3:"green", 5:"yellow", 7:"blue", 9:"red", 4:"gold",6:"green",'I':"brown"}
lss = {0:"--",1:":", 2:":",3:"-.", 4:"--", 5:"-",6:"--",7:":", 8:"-",9:"-.",'A':"--",  'S':"--",'VS':"-", 'I':"-"}

outdir = path.join(myname.base, "civ_plots/")

print("Plots at ",outdir)


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

print("Done QSO")
