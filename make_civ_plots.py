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

labels = {1:"HVEL", 3:"NOSN", 7:"DEF", 9:"FAST", 4:"WARM"}
colors = {1:"purple", 3:"green", 7:"blue", 9:"red", 4:"gold"}

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

class CIPlot(ps.CIVPlot):
    """Override functions so that total C is the default"""
    def plot_eq_width_ratio(self, color=None, ls="-", elem="C", ion=-1, line=1548):
        return ps.CIVPlot.plot_eq_width_ratio(self, color=color, ls=ls, elem=elem, ion=ion, line=line)
    def plot_covering_fraction(self, eq_thresh = 0.2, elem="C", ion=-1, line=1548):
        return ps.CIVPlot.plot_covering_fraction(self, eq_thresh = eq_thresh, elem=elem, ion=ion, line=line)
    def plot_eq_width(self, elem="C", ion=-1, line=1548):
        return ps.CIVPlot.plot_eq_width(self, elem=elem, ion=ion, line=line)
    def plot_flux_vel_offset(self, eq_thresh = 0.2, elem="C", ion=-1, line=1548):
        return ps.CIVPlot.plot_flux_vel_offset(self, eq_thresh = eq_thresh, elem=elem, ion=ion, line=line)

def do_civ_plots(name, ahalos):
    """Make a bunch of plots"""
    for ahalo in ahalos:
        ahalo.plot_eq_width_ratio()
    plt.xlabel("r perp (kpc)")
    plt.ylabel(r"$\eta_\mathrm{pair} / \eta_\mathrm{DLA}$")
    plt.ylim(0, 1)
    plt.legend()
    save_figure(path.join(outdir,name+"_CIV_eq_ratio"))
    plt.clf()

    for ahalo in ahalos:
        ahalo.plot_eq_width()
    plt.xlabel("r perp (kpc)")
    plt.ylabel(r"EW(CIV)")
    CGM_w = np.loadtxt("CGMofDLAs_avgWCIV.dat")
    plt.errorbar(CGM_w[:,0], CGM_w[:,1], yerr = CGM_w[:,2], fmt='o')
    save_figure(path.join(outdir,name+"_CIV_eq_width"))
    plt.clf()

    for ahalo in ahalos:
        ahalo.plot_covering_fraction()
    plt.xlabel("r perp (kpc)")
    plt.ylabel(r"$F(W > 0.2 \AA)$")
    CGM_c = np.loadtxt("CGMofDLAs_CfCIV.dat")
    plt.errorbar(CGM_c[:,0], CGM_c[:,2], yerr = [CGM_c[:,2]-CGM_c[:,1],CGM_c[:,3]-CGM_c[:,2]], fmt='o', xerr=[CGM_c[:,0]-[7,100,200],[100,200,300]-CGM_c[:,0]])
    plt.ylim(0,1.0)
    plt.legend()
    save_figure(path.join(outdir,name+"_CIV_coverfrac"))
    plt.clf()

    for ahalo in ahalos:
        ahalo.plot_flux_vel_offset()
    plt.xlabel("r perp (kpc)")
    plt.ylabel(r"$v_\mathrm{pair} - v_\mathrm{DLA}$ (km/s)")
    plt.legend()
    save_figure(path.join(outdir,name+"_CIV_vel_offset"))
    plt.clf()

def plot_r_offsets(ahalo):
    """Plot the positions of the sightlines relative to DLAs"""
    offsets = ahalo.get_offsets()
    rbins = [7.4,25,50,75,100,125,150,175,200,225,270]
    plt.hist(offsets,rbins, label=ahalo.label)
    plt.legend()
    save_figure(path.join(outdir,"CIV_r_offset"))
    plt.clf()

def do_other_ion_plots(name, ahalos):
    """Make equivalent widht plots for various low ion species"""
    for ahalo in ahalos:
        ahalo.plot_eq_width(label="CIV 1548")
        ahalo.plot_eq_width(color="pink", elem="C", ion=2, line=1334, label="CII 1334")
        ahalo.plot_eq_width(color="green", elem="Si", ion=2, line=1526, label="SiII 1526")
    plt.xlabel("r perp (kpc)")
    plt.ylabel(r"EW")
    CGM_w = np.loadtxt("CGMofDLAs_avgWCIV.dat")
    plt.errorbar(CGM_w[:,0], CGM_w[:,1], yerr = CGM_w[:,2], fmt='o', label="")
    plt.legend()
    save_figure(path.join(outdir,name+"_low_ion_eq_width"))
    plt.clf()

    for ahalo in ahalos:
        ahalo.plot_covering_fraction(label="CIV 1548")
        ahalo.plot_covering_fraction(color="pink", elem="C", ion=2, line=1334, label="CII 1334")
        ahalo.plot_covering_fraction(color="green", elem="Si", ion=2, line=1526, label="SiII 1526")
        ahalo.plot_covering_fraction_colden(color="black", elem="H", ion=1, label="F(LLS)")
    plt.xlabel("r perp (kpc)")
    plt.ylabel(r"$F(W > 0.2 \AA)$")
    CGM_c = np.loadtxt("CGMofDLAs_CfCIV.dat")
    plt.errorbar(CGM_c[:,0], CGM_c[:,2], yerr = [CGM_c[:,2]-CGM_c[:,1],CGM_c[:,3]-CGM_c[:,2]], fmt='o', xerr=[CGM_c[:,0]-[7,100,200],[100,200,300]-CGM_c[:,0]])
    plt.ylim(0,1.0)
    plt.legend()
    save_figure(path.join(outdir,name+"_low_ion_coverfrac"))
    plt.clf()




name = myname.get_name(5, box=10)

ahalos = []
ahalos.append(ps.CIVPlot(5, name, savefile="nr_dla_spectra.hdf5", label="SMALL"))
ahalos[0].color = "grey"
ahalos.append(CIPlot(5, name, savefile="nr_dla_spectra.hdf5", label="SMALL CALL"))
ahalos[1].color="pink"

plot_r_offsets(ahalos[0])

do_civ_plots("ion",ahalos)

do_other_ion_plots("ion", [ahalos[0],])

#Column density plots
def rel_c_colden(ahalo):
    """Column density for different carbon ions"""
    ahalo.plot_colden_ratio(color="grey",elem="C",ion=4,label="CIV")
    ahalo.plot_colden_ratio(color="pink",elem="C",ion=2, label="CII")
    ahalo.plot_colden_ratio(color="green",elem="C",ion=3, label="CIII")
    ahalo.plot_colden_ratio(color="brown",elem="C",ion=5, label="CV")
    plt.legend(loc='upper center', ncol=4)
    save_figure(path.join(outdir,"ion_C_colden_ratio"))
    plt.clf()

def rel_h_colden(ahalo):
    ahalo.plot_colden_ratio(color="grey",elem="C",ion=2,elem2="H",ion2=1, label="CII/HI")
    ahalo.plot_colden_ratio(color="pink",elem="C",ion=4,elem2="H",ion2=1, label="CIV/HI")
    plt.legend(loc='lower center', ncol=4)
    plt.yscale('log')
    save_figure(path.join(outdir,"ion_CHI_colden_ratio"))
    plt.clf()


rel_c_colden(ahalos[0])
rel_h_colden(ahalos[0])

ahalos = []
ahalos.append(ps.CIVPlot(5, name, savefile="nr_dla_spectra.hdf5"))
ahalos[0].color = "grey"
for ss in (1,3,4,7,9):
    name = myname.get_name(ss, box=25)
    ahalo = ps.CIVPlot(5, name, savefile="nr_dla_spectra.hdf5", label=labels[ss])
    ahalo.color=colors[ss]
    ahalos.append(ahalo)

do_civ_plots("feed",ahalos)

if True: #False:
    for nn in np.arange(0,7):
        gs = gridspec.GridSpec(9,2)
        ax = (plt.subplot(gs[0:4,0]), plt.subplot(gs[5:,0]), plt.subplot(gs[4,0]))
        #Adjust the default plot parameters, which do not scale well in a gridspec.
        matplotlib.rc('xtick', labelsize=8)
        matplotlib.rc('ytick', labelsize=8)
        matplotlib.rc('axes', labelsize=8)
        matplotlib.rc('font', size=6)
        matplotlib.rc('lines', linewidth=1.5)
        plot_den(ahalo, ax, nn+100, color="red")
        plot_den(ahalo, ax, nn)
        offsets = ahalo.get_offsets()
        ax[0].text(-500, 0.2,"offset (prop kpc): "+str(offsets[nn]*0.33333/0.7))
        odir = path.join(outdir, "spectra")
        save_figure(path.join(odir,str(nn)+"_cosmo"+"_CIV_spec"))
        plt.clf()
        matplotlib.rc_file_defaults()


