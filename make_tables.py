"""Load tabulated results from text files and output latex tables"""
import numpy as np
import glob
import re

def format_latex_num(number):
    """Return a strong formatting a number as, eg 3.1 x 10^4"""
    if number == 0.:
        return "$0$"
    exponent = int(np.floor(np.log10(number)))
    if 1 >= exponent > -2:
        return str("$ {0:.2f} $").format(number)
    else:
        return str("$ {0:.1f} \\times 10^{{ {1:d} }}$").format(number/10**exponent,exponent)

def load_table(txtnames, maxn=1e15, minn=1e12, colheader1 = "$N_\\mathrm{CIV}$ (cm$^{-2}$)", colheader2 = "", caption=""):
    """Load a table and output Latex"""
    if np.size(txtnames) == 1:
        txtnames = (txtnames,)
    tables = [np.loadtxt(txtname) for txtname in txtnames]
    (_,nrow) = np.shape(tables[0])
    ncol = len(txtnames)+1
    table_string = "\\begin{table} \n \\centering \n"
    table_string += "\\begin{tabular}{"+'c'*ncol+"}\n"
    table_string += "\\hline\n" + colheader1
    #Write headers
    for txtname in txtnames:
        table_string +=" & "+sim_name(txtname)+" "+colheader2
    table_string +=" \\\\ \n \hline \n"
    for row in range(nrow):
        if tables[0][0,row] > maxn*1.1 or tables[0][0,row] < minn*0.9:
            continue
        table_string+= format_latex_num(tables[0][0,row])
        for tt in tables:
            table_string+= " & "+format_latex_num(tt[1,row])
        table_string += "  \\\\ \n"
    table_string += "\\hline \n  \\end{tabular}\n "
    table_string += "\\caption{"+caption+"}\n"
    table_string += "\\label{tab:"+txtnames[0]+"}\n \\end{table}\n"
    return table_string

def sim_name(fname):
    """Get the name of the simulation from the filename"""
    match = re.search("_([479I])_[27]5", fname)
    if match.groups()[0] == '7':
        return "ILLUS 25"
    if match.groups()[0] == '4':
        return "WARM"
    if match.groups()[0] == '9':
        return "FAST"
    if match.groups()[0] == 'I':
        return "ILLUS 75"

def sim_zz(fname):
    """Get the redshift of the simulation from filename"""
    match = re.search("_([479I])_[27]5_([256])",fname)
#     if match.groups()[0] != 'I':
    if match.groups()[1] == '2':
        return "z=3.5"
    if match.groups()[1] == '5':
        return "z=2"
#     else:
#         if match.groups()[1] == '6':
#             return "z=2"
#         if match.groups()[1] == '5':
#             return "z=3.5"

def print_all_tables():
    """Print latex for all tables"""
    cddfs = glob.glob("civ_cddf_*.txt")
    caption = "CIV CDDF $f(N)$ in cm$^2$ for our simulations"
    print(load_table(cddfs, maxn = 1e15, minn=1e12, colheader2 = "",caption = caption))
    eqws = glob.glob("eq_width_*_5.txt")
    caption = "Equivalent width distributions , $f(W_{1548})$ (\AA$^{-1}$), for our simulations at "+sim_zz(eqws[0])
    print(load_table(eqws, maxn = 2, minn=0.04, colheader1 = "$W_{1548}$ (\\AA) ", colheader2 = "" ,caption=caption))
    eqws = glob.glob("eq_width_*_2.txt")
    caption = "Equivalent width distributions , $f(W_{1548})$ (\AA$^{-1}$), for our simulations at "+sim_zz(eqws[0])
    print(load_table(eqws, maxn = 2, minn=0.04, colheader1 = "$W_{1548}$ (\\AA) ", colheader2 = "" ,caption=caption))
    lcs = glob.glob("lciv_06*.txt")
    caption = "dN/dX ( $W_{1548} > 0.6$\,\AA) for our simulations"
    print(load_table(lcs, maxn = 4, minn=2, colheader1 = "$z$", colheader2  = "" ,caption=caption))
    omega_civ = glob.glob("om_civ_*.txt")
    caption = "$\\Omega_\\mathrm{CIV} \\times 10^{8}$ for our simulations"
    print(load_table(omega_civ, maxn = 4, minn=2, colheader1 = "$z$", colheader2  = "",caption=caption ))

