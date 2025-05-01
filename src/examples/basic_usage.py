from wingfea import setupParams, wingMesher, prepMesh, elasticityTensor
from wingfea.solver import elementMatrices, assembleGlobalSys, boundaryConditions, stress
from wingfea.visualization import plotSparsities, setupParaviewData, vtkEigenAnalysis


def main():
    # Define problem parameters
    E1, E2, nu12, G12, quadNo, L1, theta, phi, area, meshSize, nne, force, eigshModes, filename = setupParams()
    Ce = elasticityTensor(E1, E2, nu12, G12)

    # Mesh generation and preprocessing
    meshioPy, con_mat, nodeCoor, nelem, ndof, qpt, qwt, dofpe, nnodes = prepMesh(
        filename, L1, theta, phi, area, meshSize, nne, quadNo
    )

    # Initialising & Populating Elemental Arrays
    K, kEl, aEl, BEl, FwEl = elementMatrices(nelem, dofpe, ndof, nodeCoor, con_mat, qpt, qwt, Ce, nne)

    # Assemble global stiffness matrix
    K = assembleGlobalSys(K, kEl, con_mat)

    # Global stiffness matrix w/ boundary conditions, and displacements
    K_bc, u, ux, uy = boundaryConditions(nodeCoor, K, ndof, force, FwEl, con_mat)

    # Calculate stresses
    sigma1, sigma2, shear, vm, shear1 = stress(u, con_mat, nelem, BEl, Ce)

    # Setup Mesh for Visualisation
    meshioPy, deformedMesh = setupParaviewData(meshioPy, ux, uy, nodeCoor, nelem, sigma1, sigma2, vm, shear, shear1,
                                               aEl)

    # Use Spy to plot sparsity matrices
    plotSparsities(K, K_bc)

    # Calculate Eigenvalues and vectors for further analysis
    finalMesh = vtkEigenAnalysis(K_bc, eigshModes, meshioPy)

    # Save the mesh
    deformedMesh.save(filename + "Deformed.vtk", binary=False)
    finalMesh.save(filename + ".vtk", binary=False)


if __name__ == '__main__':
    main()