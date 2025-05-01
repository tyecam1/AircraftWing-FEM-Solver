import numpy as np
import scipy as sp
import scipy.sparse
from .utils import jacobian, strainDisp2D
from thirdParty.shapeFun import shapeFun

def quadCalcs(qpt, qwt, xyel, Ce):
    """Calculates the element stiffness matrix for plane stress."""
    nne = xyel.shape[0]  # Number of nodes per element
    ndim = xyel.shape[1]  # Number of spatial dimensions (2)
    nstress = Ce.shape[0]  # Number of stress components
    ndof = nne * ndim  # Number of degrees of freedom

    # Create arrays of quadrature points and weights
    xi, eta = np.meshgrid(qpt, qpt)
    weight = np.outer(qwt, qwt).flatten()
    xi = xi.flatten()
    eta = eta.flatten()

    npts = xi.shape[0]  # Total number of integration points (nquad^2)

    # Initialize arrays to store computed values
    Bemats = np.zeros((npts, nstress, ndof))
    kemats = np.zeros((npts, ndof, ndof))
    detJs = np.zeros(npts)

    # Loop over all quadrature points
    for i in range(npts):
        Xi = xi[i]
        Eta = eta[i]

        # Compute shape functions and their derivatives
        SF, dndxi, dndeta = shapeFun(Xi, Eta, nne)  # Shape function & derivatives vectors (nne,)

        # Compute Jacobian and its determinant
        J, invJ, detJ = jacobian((dndxi, dndeta), xyel)

        # Compute strain-displacement matrix
        Bemat = strainDisp2D({'sf': SF, 'dndx': dndxi, 'dnde': dndeta}, xyel,
                             {'J': J, 'invJ': invJ, 'detJ': detJ})

        # Compute element stiffness matrix at this quadrature point
        kemat = Bemat.T @ Ce @ Bemat

        # Store computed values
        Bemats[i, :, :] = Bemat
        kemats[i, :, :] = kemat
        detJs[i] = detJ

    # Compute the factors for integration (weights multiplied by determinants)
    factors = weight * detJs

    # Integrate over the element by summing contributions from all quadrature points
    Be = np.einsum('i,ijk->jk', factors, Bemats)  # Element strain displacement matrix
    ke = np.einsum('i,ijk->jk', factors, kemats)  # Element stiffness matrix
    Ae = np.sum(factors)  # Area of element
    Fe = 1 / (Ae * nne)  # Force pressure factor of element
    return ke, Ae, Be, Fe

def assembleGlobalSys(K, Kel, con_el):
    """Assembles the global stiffness matrix."""
    # Determine degrees of freedom per element (dofpe) from the shape of a local stiffness matrix
    dofpe = Kel[0].shape[0]  # Local stiffness matrix is square, so dofpe = rows = cols

    # Get the number of nodes per element from the connectivity matrix
    npe = con_el.shape[1]  # Number of nodes per element

    # Degrees of freedom per node (dofpn), computed as total DOFs per element divided by nodes per element
    dofpn = dofpe / npe

    # Total number of global degrees of freedom in the system
    ndof = K.shape[0]

    # Total number of elements in the system
    nelem = con_el.shape[0]

    # Loop through each element to assemble its contribution to the global stiffness matrix
    for ii in range(nelem):
        # Get the global node indices for the current element from the connectivity matrix
        # Compute the global row indices (Rind) for the current element
        # Each node contributes multiple degrees of freedom, calculated as dofpn * ix
        Rind = np.c_[dofpn * con_el[ii], dofpn * con_el[ii] + 1].flatten()

        # Compute the local column indices for the current element
        Cind = np.arange(dofpe)  # Local DOF indices are sequential for each element stiffness matrix

        # Create a vector of ones for sparse matrix construction
        indV = np.ones(dofpe)

        # Convert the local stiffness matrix (Kel[ii]) to a sparse matrix for efficient computation
        Kelsparse = sp.sparse.csr_matrix(Kel[ii])

        # Create the sparse transformation matrix (Emat) to map local to global DOFs
        Emat = sp.sparse.csr_matrix((indV, (Rind, Cind)), shape=(ndof, dofpe))

        # Add the transformed local stiffness matrix contribution to the global stiffness matrix
        # The dot product calculates the mapping and accumulates the contribution
        K = K + Emat.dot(Kelsparse).dot(Emat.T)
    return K

def elementMatrices(nelem, dofpe, ndof, nodeCoor, con_mat, qpt, qwt, Ce, nne):
    """Initializes and populates elemental matrices."""
    kEl = np.zeros([nelem, dofpe, dofpe])  # stiffness matrix for each element
    aEl = np.zeros(nelem)  # empty matrix for area per element and element strain displacement matrix
    BEl = np.zeros([nelem, 3, dofpe])
    FwEl = np.zeros(nelem)
    K = sp.sparse.csr_matrix((ndof, ndof))  # initialise sparse k matrix
    for el in range(nelem):
        xyel = nodeCoor[con_mat[el], :][:, :2]  # extracting the xy co-ords for each node
        kEl[el], aEl[el], BEl[el], FwEl[el] = quadCalcs(qpt, qwt, xyel,
                                                        Ce)  # populating stiffness, area, strain displacement and weighted force matrices for each element
    return K, kEl, aEl, BEl, FwEl

def boundaryConditions(nodeCoor, K, ndof, force, FwEl, con_mat):
    """Applies boundary conditions and solves for displacements."""
    ##### Dirichlet BC (edge x=0) #######
    tol = 1e-8  # Tolerance for floating-point comparisons
    # Find nodes where the x coordinates have their minimum value
    nodesbc = np.where(np.abs(nodeCoor[:, 0] - nodeCoor[:, 0].min()) < tol)[0]
    # Flatten the degrees of freedom where the nodes are restricted, and concatenate the x and y coordinates
    dofbc = np.c_[2 * nodesbc, 2 * nodesbc + 1].flatten()
    K_bc = DirichletBC(K, dofbc)  # System matrix after boundary conditions
    K_bc = K_bc.tocsr()  # Dirichlet BCs function has been edited to use lil_matrix for maximum efficiency

    # Force
    F = np.zeros(ndof)  # empty array of forces for each degree of freedom
    F[2 * con_mat + 1] += (force * FwEl[:, None])  # apply drag force to y dofs

    # Strain
    u = sp.sparse.linalg.spsolve(K_bc, F)  # solve for displacements
    ux = u[::2]  # extract x
    uy = u[1::2]  # extract y

    return K_bc, u, ux, uy

def stress(u, con_mat, nelem, B, D):
    """Computes stress components for each element."""
    # Initialize stress arrays for all elements
    Sigma1 = np.zeros((nelem, 1))  # Normal stress in x-direction
    Sigma2 = np.zeros((nelem, 1))  # Normal stress in y-direction
    Shear = np.zeros((nelem, 1))  # Shear stress
    Shear1 = np.zeros((nelem, 1))  # Maximum shear stress (principal shear stress)
    VM = np.zeros((nelem, 1))  # Von Mises stress

    # Loop through each element to calculate stress components
    for ii in range(nelem):
        # Get global node indices for the current element from the connectivity matrix
        ix = con_mat[ii]

        # Calculate global degrees of freedom (DOFs) for the current element
        dofix = np.c_[2 * ix, 2 * ix + 1].flatten()  # Each node contributes 2 DOFs (x and y displacements)

        # Extract the element's displacement vector from the global displacement vector
        u_elem = u[dofix]

        # Compute the stress vector (σ = D * B * u) for the element
        D_B = D.dot(B[ii])  # Precompute D * B matrix product
        Sigma_E = D_B.dot(u_elem)  # Stress components for the current element

        # Principal stresses calculation
        # P1 and P2 are the maximum and minimum principal stresses, respectively
        P1 = 0.5 * (Sigma_E[0] + Sigma_E[1]) + np.sqrt((0.5 * (Sigma_E[0] - Sigma_E[1])) ** 2 + Sigma_E[2] ** 2)
        P2 = 0.5 * (Sigma_E[0] + Sigma_E[1]) - np.sqrt((0.5 * (Sigma_E[0] - Sigma_E[1])) ** 2 + Sigma_E[2] ** 2)

        # Maximum shear stress is half the difference between principal stresses
        Shear1[ii] = 0.5 * (P1 - P2)

        # Assign normal and shear stresses from the computed stress vector
        Sigma1[ii] = Sigma_E[0]  # Normal stress in x-direction
        Sigma2[ii] = Sigma_E[1]  # Normal stress in y-direction
        Shear[ii] = Sigma_E[2]  # Shear stress

        # Von Mises stress calculation (used for failure prediction in materials)
        # VM = sqrt((P1 - P2)^2 + P2^2 + P1^2) / sqrt(2)
        VM[ii] = (1 / np.sqrt(2)) * np.sqrt((P1 - P2) ** 2 + P2 ** 2 + P1 ** 2)

    # Return all calculated stress components
    return Sigma1, Sigma2, Shear, VM, Shear1