#!/usr/bin env python
# -*- coding: utf-8 -*-
"""Make some plots of the velocity widths from the cosmo runs"""

import matplotlib
matplotlib.use('PDF')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import plot_spectra as ps
import os.path as path
import myname
import numpy as np
from save_figure import save_figure

outdir = path.join(myname.base, "civ_plots/")

def plot_den(hspec, ax, num, voff = 0, elem="C", ion=4, color="blue"):
    """Plot density"""
    plt.sca(ax[0])
    xoff = hspec.plot_spectrum(elem,ion,1548,num, flux=False, color=color)
    #ax[0].axes.set_yscale('log')
    xlim = plt.xlim()
    plt.ylim(0,15)
    ax[0].xaxis.set_label_position('top')
    ax[0].xaxis.tick_top()
    voff += xoff
    plt.sca(ax[1])
    dxlim = hspec.plot_density(elem,ion, num, color=color)
    plt.ylabel(r"n$_\mathrm{"+elem+"IV}$ (cm$^{-3}$)")
    #plt.ylim(ymin=1e-9)
    plt.sca(ax[2])
    xscale = dxlim*hspec.velfac/xlim[1]
    #hspec.plot_den_to_tau(elem, ion, num, thresh = 1e-9, xlim=200,voff=voff, xscale=xscale)
    ax[2].axes.get_xaxis().set_visible(False)
    plt.xlabel("")
    plt.xlim(xlim)

name = myname.get_name(5, box=10)

ahalo = ps.PlottingSpectra(5, name, savefile="nr_dla_spectra.hdf5")

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
    ax[0].text(-500, 3,"offset: "+str(np.floor(np.sqrt(np.sum((ahalo.cofm[nn,:] - ahalo.cofm[nn+100,:])**2,axis=1)))))
    save_figure(path.join(outdir,str(nn)+"_cosmo"+"_CIV_spec"))
    plt.clf()
    matplotlib.rc_file_defaults()

