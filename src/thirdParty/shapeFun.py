import numpy as np

def shapeFun(xi, eta, nne):
    """
    Calculates the shape function at specified points in the local
    coordinate specified by xi and eta.

    Inputs  :
    xi, eta : local coordinates in the quadrangular element
    nne     : number of nodes per element (scalar)

    Outputs :
    sn      : vector of shape functions (1 by nne)
    dndx    : vector derivatives of shape functions w.r.t xi (1 by nne)
    dnde    : vector derivatives of shape functions w.r.t eta (1 by nne)
    """
    # case for 4 nodes per element (linear)
    if nne == 4:
        sn = np.zeros(4)
        sn[0] = (1.0 - xi) * (1.0 - eta) / 4.0
        sn[1] = (1.0 + xi) * (1.0 - eta) / 4.0
        sn[2] = (1.0 + xi) * (1.0 + eta) / 4.0
        sn[3] = (1.0 - xi) * (1.0 + eta) / 4.0

        dndx, dnde = np.zeros(4), np.zeros(4)
        dndx[0] = -(1.0 - eta) / 4.0
        dnde[0] = -(1.0 - xi) / 4.0
        dndx[1] = (1.0 - eta) / 4.0
        dnde[1] = -(1.0 + xi) / 4.0
        dndx[2] = (1.0 + eta) / 4.0
        dnde[2] = (1.0 + xi) / 4.0
        dndx[3] = -(1.0 + eta) / 4.0
        dnde[3] = (1.0 - xi) / 4.0

    # case for 8 nodes per element (quadratic)
    elif nne == 8:
        sn = np.zeros(8)
        sn[0] = -(1.0 - xi) * (1.0 - eta) * (1.0 + xi + eta) / 4.0
        sn[1] = (1.0 - xi ** 2) * (1.0 - eta) / 2.0
        sn[2] = (1.0 + xi) * (1.0 - eta) * (-1.0 + xi - eta) / 4.0
        sn[3] = (1.0 + xi) * (1.0 - eta ** 2) / 2.0
        sn[4] = (1.0 + xi) * (1.0 + eta) * (-1.0 + xi + eta) / 4.0
        sn[5] = (1.0 - xi ** 2) * (1.0 + eta) / 2.0
        sn[6] = (1.0 - xi) * (1.0 + eta) * (-1.0 - xi + eta) / 4.0
        sn[7] = (1.0 - xi) * (1.0 - eta ** 2) / 2.0

        dndx, dnde = np.zeros(8), np.zeros(8)
        dndx[0] = -((eta - 1.0) * (2.0 * xi + eta)) / 4.0
        dnde[0] = -((xi - 1.0) * (xi + 2.0 * eta)) / 4.0
        dndx[1] = (eta - 1.0) * xi
        dnde[1] = ((xi - 1.0) * (xi + 1.0)) / 2.0
        dndx[2] = -((eta - 1.0) * (2.0 * xi - eta)) / 4.0
        dnde[2] = -((xi + 1.0) * (xi - 2.0 * eta)) / 4.0
        dndx[3] = -((eta - 1.0) * (eta + 1.0)) / 2.0
        dnde[3] = -eta * (xi + 1.0)
        dndx[4] = ((eta + 1.0) * (2.0 * xi + eta)) / 4.0
        dnde[4] = ((xi + 1.0) * (xi + 2.0 * eta)) / 4.0
        dndx[5] = -(eta + 1.0) * xi
        dnde[5] = -((xi - 1.0) * (xi + 1.0)) / 2.0
        dndx[6] = ((eta + 1.0) * (2.0 * xi - eta)) / 4.0
        dnde[6] = ((xi - 1.0) * (xi - 2.0 * eta)) / 4.0
        dndx[7] = ((eta - 1.0) * (eta + 1.0)) / 2.0
        dnde[7] = eta * (xi - 1.0)

    else:
        raise ValueError("Unsupported number of nodes per element (", nne, ". Use nne=3, nne=4, nne=6, or nne=8.")

    return sn, dndx, dnde