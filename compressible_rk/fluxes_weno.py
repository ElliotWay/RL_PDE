"""
This is a 2nd-order PLM method for a method-of-lines integration
(i.e., no characteristic tracing).

We wish to solve

.. math::

   U_t + F^x_x + F^y_y = H

we want U_{i+1/2} -- the interface values that are input to
the Riemann problem through the faces for each zone.

Taylor expanding *in space only* yields::

                              dU
   U          = U   + 0.5 dx  --
    i+1/2,j,L    i,j          dx

"""

import compressible.interface as interface
import compressible as comp
import mesh.reconstruction as reconstruction
import mesh.array_indexer as ai
import compressible_rk.weno_coefficients as weno_coefficients
import numpy
from util.misc import create_stencil_indexes
from envs.weno_solution import weno_sub_stencils_nd

from util import msg

def weno_ii(order, nvar, q):
    """
    Do WENO reconstruction at a given location

    Parameters
    ----------

    order : int
        The stencil width
    q : numpy array
        Scalar data to reconstruct

    Returns
    -------

    qL : float
        Reconstructed data - boundary points are zero
    """
    C = weno_coefficients.C_all[order]
    a = weno_coefficients.a_all[order]
    sigma = weno_coefficients.sigma_all[order]

    qL = numpy.zeros(nvar)
    beta = numpy.zeros((order))
    w = numpy.zeros_like(beta)
    epsilon = 1e-16

    for nv in range(nvar):
        q_stencils = numpy.zeros(order)
        alpha = numpy.zeros(order)
        for k in range(order):
            for l in range(order):
                for m in range(l + 1):
                    beta[k] += sigma[k, l, m] * q[nv, order - 1 + k - l] * q[nv, order - 1 + k - m]
            # alpha[k] = C[k] / (epsilon + beta[k] ** 2)
            alpha[k] = C[k] / (epsilon + abs(beta[k]) ** order)  # TODO: check this!
            for l in range(order):
                q_stencils[k] += a[k, l] * q[nv, order - 1 + k - l]
        w[:] = alpha / numpy.sum(alpha)
        qL[nv] = numpy.dot(w[:], q_stencils)

    return qL

def weno_states(idir, ng, qv, weno_order=2, agent=None):
    
    qx, qy, nvar = qv.shape

    q_l = numpy.zeros_like(qv)
    q_r = numpy.zeros_like(qv)
    
    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny
    w_o = weno_order
    
    for i in range(ilo - 2, ihi + 2):
        for j in range(jlo - 2, jhi + 2):
            if (idir == 1): # x-direction
                # q_l[i, j, :] = weno_ii(w_o, nvar, qv[i - w_o: i + w_o - 1, j, :].T)
                # q_r[i, j, :] = weno_ii(w_o, nvar, numpy.flip(qv[i - (w_o - 1):i + w_o, j, :], 0).T) #flip the x-direction
                # print(qv[i - w_o: i + w_o - 1, j, :], q_l[i,j,:])
                # print(qv[i - (w_o - 1):i + w_o, j, :], q_r[i, j, :])

                fp_stencil_indexes = create_stencil_indexes(stencil_size=weno_order * 2 - 1,
                                                            num_stencils=1,
                                                            offset=i-ilo+3)
                fm_stencil_indexes = fp_stencil_indexes + 1
                fp_stencils = [qv[:, j, :].T[ii][fp_stencil_indexes] for ii in range(nvar)]
                fm_stencils = [numpy.flip(qv[:, j, :], 0).T[ii][fm_stencil_indexes] for ii in range(nvar)]

                fm_stencils = numpy.flip(fm_stencils, axis=-1)
                fp_stencils = numpy.stack(fp_stencils, axis=0)
                state = numpy.array([fp_stencils, fm_stencils])
                fp_substencils = weno_sub_stencils_nd(fp_stencils, weno_order)
                fm_substencils = weno_sub_stencils_nd(fm_stencils, weno_order)

                state = state.transpose((1, 2, 0, 3))
                weights, _ = agent.predict(state, deterministic=True)
                fp_weights = weights[:, :, 0, :]
                fm_weights = weights[:, :, 1, :]
                q_l[i, j, :] = numpy.squeeze(numpy.sum(fp_weights * fp_substencils, axis=-1))
                q_r[i, j, :] = numpy.squeeze(numpy.sum(fm_weights * fm_substencils, axis=-1))
            else:
                # q_l[i, j, :] = weno_ii(w_o, nvar, qv[i, j - w_o: j + w_o - 1, :].T)
                # q_r[i, j, :] = weno_ii(w_o, nvar, numpy.flip(qv[i, j - (w_o - 1):j + w_o, :], 0).T) #flip the y-direction

                fp_stencil_indexes = create_stencil_indexes(stencil_size=weno_order * 2 - 1,
                                                            num_stencils=1,
                                                            offset=j-jlo+3)
                fm_stencil_indexes = fp_stencil_indexes + 1
                fp_stencils = [qv[i:, ii, :].T[ii][fp_stencil_indexes] for ii in range(nvar)]
                fm_stencils = [numpy.flip(qv[i:, ii, :], 0).T[ii][fm_stencil_indexes] for ii in range(nvar)]

                fm_stencils = numpy.flip(fm_stencils, axis=-1)
                fp_stencils = numpy.stack(fp_stencils, axis=0)
                state = numpy.array([fp_stencils, fm_stencils])
                fp_substencils = weno_sub_stencils_nd(fp_stencils, weno_order)
                fm_substencils = weno_sub_stencils_nd(fm_stencils, weno_order)

                state = state.transpose((1, 2, 0, 3))
                weights, _ = agent.predict(state, deterministic=True)
                fp_weights = weights[:, :, 0, :]
                fm_weights = weights[:, :, 1, :]
                q_l[i, j, :] = numpy.squeeze(numpy.sum(fp_weights * fp_substencils, axis=-1))
                q_r[i, j, :] = numpy.squeeze(numpy.sum(fm_weights * fm_substencils, axis=-1))

    return q_l, q_r

def fluxes(my_data, rp, ivars, solid, tc, agent=None):
    """
    unsplitFluxes returns the fluxes through the x and y interfaces by
    doing an unsplit reconstruction of the interface values and then
    solving the Riemann problem through all the interfaces at once

    currently we assume a gamma-law EOS

    Parameters
    ----------
    my_data : CellCenterData2d object
        The data object containing the grid and advective scalar that
        we are advecting.
    rp : RuntimeParameters object
        The runtime parameters for the simulation
    vars : Variables object
        The Variables object that tells us which indices refer to which
        variables
    tc : TimerCollection object
        The timers we are using to profile

    Returns
    -------
    out : ndarray, ndarray
        The fluxes on the x- and y-interfaces

    """

    tm_flux = tc.timer("unsplitFluxes")
    tm_flux.begin()

    myg = my_data.grid

    gamma = rp.get_param("eos.gamma")
    weno_order = rp.get_param("compressible.weno_order")

    q = my_data.data

    # =========================================================================
    # x-direction
    # =========================================================================

    # left and right primitive variable states
    tm_states = tc.timer("interfaceStates")
    tm_states.begin()
    U_xl, U_xr = weno_states(1, myg.ng, q, weno_order=weno_order, agent=agent)
    tm_states.end()

    # =========================================================================
    # y-direction
    # =========================================================================

    # left and right primitive variable states
    tm_states.begin()
    U_yl, U_yr = weno_states(2, myg.ng, q, weno_order=weno_order, agent=agent)
    tm_states.end()


    # =========================================================================
    # construct the fluxes normal to the interfaces
    # =========================================================================
    tm_riem = tc.timer("Riemann")
    tm_riem.begin()

    riemann = rp.get_param("compressible.riemann")

    if riemann == "HLLC":
        riemannFunc = interface.riemann_hllc
    elif riemann == "CGF":
        riemannFunc = interface.riemann_cgf
    else:
        msg.fail("ERROR: Riemann solver undefined")

    # FIXME: hard coded to do HLLC
    riemannFunc = interface.riemann_hllc
    _fx = riemannFunc(1, myg.ng,
                      ivars.idens, ivars.ixmom, ivars.iymom, ivars.iener, ivars.irhox, ivars.naux,
                      solid.xl, solid.xr,
                      gamma, U_xl, U_xr)

    _fy = riemannFunc(2, myg.ng,
                      ivars.idens, ivars.ixmom, ivars.iymom, ivars.iener, ivars.irhox, ivars.naux,
                      solid.yl, solid.yr,
                      gamma, U_yl, U_yr)

    F_x = ai.ArrayIndexer(d=_fx, grid=myg)
    F_y = ai.ArrayIndexer(d=_fy, grid=myg)

    tm_riem.end()

    # =========================================================================
    # apply artificial viscosity - TURN OFF ?? FIXME
    # =========================================================================
    # cvisc = rp.get_param("compressible.cvisc")
    #
    # _ax, _ay = interface.artificial_viscosity(myg.ng, myg.dx, myg.dy,
    #     cvisc, q.v(n=ivars.iu, buf=myg.ng), q.v(n=ivars.iv, buf=myg.ng))
    #
    # avisco_x = ai.ArrayIndexer(d=_ax, grid=myg)
    # avisco_y = ai.ArrayIndexer(d=_ay, grid=myg)
    #
    # b = (2, 1)
    #
    # for n in range(ivars.nvar):
    #     # F_x = F_x + avisco_x * (U(i-1,j) - U(i,j))
    #     var = my_data.get_var_by_index(n)
    #
    #     F_x.v(buf=b, n=n)[:, :] += \
    #         avisco_x.v(buf=b) * (var.ip(-1, buf=b) - var.v(buf=b))
    #
    #     # F_y = F_y + avisco_y * (U(i,j-1) - U(i,j))
    #     F_y.v(buf=b, n=n)[:, :] += \
    #         avisco_y.v(buf=b) * (var.jp(-1, buf=b) - var.v(buf=b))

    tm_flux.end()

    return F_x, F_y
