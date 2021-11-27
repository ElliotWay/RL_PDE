from __future__ import print_function

import sys
import numpy
import mesh.patch as patch
from util import msg


def init_data(my_data, rp):
    """ initialize the Gresho vortex problem """

    msg.bold("initializing the Gresho vortex problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ErrrrOrr: patch invalid in bubble.py")
        print(my_data.__class__)
        sys.exit()

    # get the density and velocities
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    # grav = rp.get_param("compressible.grav")

    gamma = rp.get_param("eos.gamma")

    # scale_height = rp.get_param("gresho.scale_height")
    dens_base = rp.get_param("gresho.dens_base")
    dens_cutoff = rp.get_param("gresho.dens_cutoff")

    rr = rp.get_param("gresho.r")
    u0 = rp.get_param("gresho.u0")
    p0 = rp.get_param("gresho.p0")

    # initialize the components -- we'll get a psure too
    # but that is used only to initialize the base state
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0
    dens[:, :] = dens_cutoff

    # set the density to be stratified in the y-direction
    myg = my_data.grid
    pres = myg.scratch_array()

    dens[:, :] = dens_base

    pres[:, :] = p0

    x_centre = 0.5 * (myg.x[0] + myg.x[-1])
    y_centre = 0.5 * (myg.y[0] + myg.y[-1])

    rad = numpy.sqrt((myg.x2d - x_centre)**2 + (myg.y2d - y_centre)**2)

    pres[rad <= rr] += 0.5 * (u0 * rad[rad <= rr]/rr)**2
    pres[(rad > rr) & (rad <= 2*rr)] += \
        u0**2 * (0.5 * (rad[(rad > rr) & (rad <= 2*rr)] / rr)**2 +
        4 * (1 - rad[(rad > rr) & (rad <= 2*rr)]/rr +
        numpy.log(rad[(rad > rr) & (rad <= 2*rr)]/rr)))
    pres[rad > 2*rr] += u0**2 * (4 * numpy.log(2) - 2)
    #
    uphi = numpy.zeros_like(pres)
    uphi[rad <= rr] = u0 * rad[rad <= rr]/rr
    uphi[(rad > rr) & (rad <= 2*rr)] = u0 * (2 - rad[(rad > rr) & (rad <= 2*rr)]/rr)

    xmom[:, :] = -dens[:, :] * uphi[:, :] * (myg.y2d - y_centre) / rad[:, :]
    ymom[:, :] = dens[:, :] * uphi[:, :] * (myg.x2d - x_centre) / rad[:, :]

    ener[:, :] = pres[:, :]/(gamma - 1.0) + \
                0.5*(xmom[:, :]**2 + ymom[:, :]**2)/dens[:, :]

    eint = pres[:, :]/(gamma - 1.0)

    dens[:, :] = pres[:, :]/(eint[:, :]*(gamma - 1.0))


def finalize():
    """ print out any information to the userad at the end of the run """
    pass
