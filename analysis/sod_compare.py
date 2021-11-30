#!/usr/bin/env python3

from __future__ import print_function

import numpy as np
from util import io
import sys
import matplotlib.pyplot as plt
import os

# usage = """
#       compare the output for a Sod problem with the exact solution contained
#       in sod-exact.out.
#
#       usage: ./sod_compare.py file
# """
#
#
# def abort(string):
#     print(string)
#     sys.exit(2)
#
#
# if not len(sys.argv) == 2:
#     print(usage)
#     sys.exit(2)
#
# try:
#     file1 = sys.argv[1]
# except IndexError:
#     print(usage)
#     sys.exit(2)

file1 = '../log/pyro_2/sod_order_2/sod_x_0138.h5'

sim = io.read(file1)
dir_name = os.path.dirname(file1)
myd = sim.cc_data
myg = myd.grid

dens = myd.get_var("density")
xmom = myd.get_var("x-momentum")
ymom = myd.get_var("y-momentum")
ener = myd.get_var("energy")

# get the exact solution
exact = np.loadtxt("sod-exact.out")

x_exact = exact[:, 0]
rho_exact = exact[:, 1]
u_exact = exact[:, 2]
p_exact = exact[:, 3]
e_exact = exact[:, 4]

# get the 1-d profile from the simulation data -- assume that whichever
# coordinate is the longer one is the direction of the problem
if myg.nx > myg.ny:
    # x-problem
    x = myg.x[myg.ilo:myg.ihi+1]
    jj = myg.ny//2

    rho = dens[myg.ilo:myg.ihi+1, jj]

    u = xmom[myg.ilo:myg.ihi+1, jj]/rho
    ut = ymom[myg.ilo:myg.ihi+1, jj]/rho

    e = (ener[myg.ilo:myg.ihi+1, jj] - 0.5*rho*(u*u + ut*ut))/rho

    gamma = myd.get_aux("gamma")
    p = rho*e*(gamma - 1.0)

else:

    # y-problem
    x = myg.y[myg.jlo:myg.jhi+1]
    ii = myg.nx//2

    rho = dens[ii, myg.jlo:myg.jhi+1]

    u = ymom[ii, myg.jlo:myg.jhi+1]/rho
    ut = xmom[ii, myg.jlo:myg.jhi+1]/rho

    e = (ener[ii, myg.jlo:myg.jhi+1] - 0.5*rho*(u*u + ut*ut))/rho

    gamma = myd.get_aux("gamma")
    p = rho*e*(gamma - 1.0)


print(myg)

# plot
fig, axes = plt.subplots(nrows=4, ncols=1, num=1)

plt.rc("font", size=10)


ax = axes.flat[0]
lw=0.5
ax.plot(x_exact, rho_exact, 'k--', linewidth = lw)
ax.plot(x, rho, 'b-', linewidth = lw) #marker="x", s=7, color="r")

ax.set_ylabel(r"$\rho$")
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.1)

ax = axes.flat[1]

ax.plot(x_exact, u_exact, 'k--', linewidth = lw)
ax.plot(x, u, 'b-', linewidth = lw) # marker="x", s=7, color="r")

ax.set_ylabel(r"$u$")
ax.set_xlim(0, 1.0)

ax = axes.flat[2]

ax.plot(x_exact, p_exact, 'k--', linewidth = lw)
ax.plot(x, p, 'b-', linewidth = lw) # marker="x", s=7, color="r")

ax.set_ylabel(r"$p$")
ax.set_xlim(0, 1.0)

ax = axes.flat[3]

ax.plot(x_exact, e_exact, 'k--', linewidth = lw)
ax.plot(x, e, 'b-', linewidth = lw) # marker="x", s=7, color="r")

if (myg.nx > myg.ny):
    ax.set_xlabel(r"x")
else:
    ax.set_xlabel(r"y")

ax.set_ylabel(r"$e$")
ax.set_xlim(0, 1.0)

plt.subplots_adjust(hspace=0.25)

fig.set_size_inches(4.5, 9.0)

plt.savefig(os.path.join(dir_name, "sod_compare.pdf"), bbox_inches="tight")
