"""Little module to find the path of a Cosmo box simulation"""

import os.path as path

base=path.expanduser("~/data/Cosmo/")

def get_name(sim, ff=True, box=25):
    """Get the directory for a simulation"""
    if sim == 'I' and box==75:
        return path.expanduser("~/data/Illustris")
    halo = "Cosmo"+str(sim)+"_V6"
    if ff:
        halo=path.join(halo,"L"+str(box)+"n512/output")
    else:
        halo=path.join(halo,"L"+str(box)+"n256")
    return path.join(base, halo)
