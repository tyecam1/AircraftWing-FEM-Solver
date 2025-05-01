import matplotlib.pyplot as plt
import pyvista as pv
import scipy as sp

def plotSparsities(K, K_bc):
    """Plots sparsity patterns of stiffness matrices."""
    # plot stiffness before boundary conditions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spy(K, markersize=0.1)
    ax.grid(False)
    plt.xlabel(r'columns')
    plt.ylabel(r'rows')
    flnmfig = "sparsity_K.png"
    plt.savefig(flnmfig)

    # plot stiffness after conditions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spy(K_bc, markersize=0.1)
    ax.grid(False)
    plt.xlabel(r'columns')
    plt.ylabel(r'rows')
    plt.show()  # uncomment to show the plot
    flnmfig = "sparsity_K_afterBC.png"
    plt.savefig(flnmfig)

def setupParaviewData(meshioPy, ux, uy, nodeCoor, nelem, sigma1, sigma2, vm, shear, shear1, aEl):
    """Sets up data for visualization in ParaView."""
    # Add distance from the origin dataset to the point data (calculated from the coordinates)
    meshioPy.point_data['Distance from Origin'] = np.sqrt(np.sum((meshioPy.points ** 2), axis=1))

    # Add element number dataset for visualization (calculated from the cell data)
    meshioPy.cell_data['Element Number '] = np.sqrt(np.arange(meshioPy.n_cells))

    # Add displacement (ux, uy) to point data for visualization, assuming zero displacement in z-direction
    meshioPy.point_data['Displacement Magnitude'] = np.c_[ux, uy, np.zeros_like(uy)]

    # Calculate deformed points by adding displacement to the original coordinates
    deformedPoints = meshioPy.point_data['Displacement Magnitude'] + nodeCoor

    # Create a copy of the mesh to represent the deformed mesh and update the points
    deformedMesh = meshioPy.copy()
    deformedMesh.points = deformedPoints

    # Add individual displacement components (ux, uy) to the deformed mesh's point data
    deformedMesh.point_data['Ux'] = ux
    deformedMesh.point_data['Uy'] = uy

    # Calculate the index where quadrangular cells start in the mesh (quad cell index is the index of meta data)
    quadCellInd = meshioPy.n_cells - nelem

    # Initialize stress and other element data with minimum values to avoid missing data
    meshioPy.cell_data['Sigma xx'] = np.ones(meshioPy.n_cells) * sigma1.min()
    meshioPy.cell_data['Sigma yy'] = np.ones(meshioPy.n_cells) * sigma2.min()
    meshioPy.cell_data['Von Mises'] = np.ones(meshioPy.n_cells) * vm.min()
    meshioPy.cell_data['Shear Stress'] = np.ones(meshioPy.n_cells) * shear.min()
    meshioPy.cell_data['Area per Element'] = np.ones(
        meshioPy.n_cells) * aEl.min()  # Element areas (assuming this is the correct field name)
    meshioPy.cell_data['Maximum Principle Shear Stress'] = np.ones(meshioPy.n_cells) * shear1.min()

    # Assign actual data to the corresponding elements in the mesh (for quadrangular elements)
    meshioPy.cell_data['Area per Element'][quadCellInd:] = aEl
    meshioPy.cell_data['Sigma xx'][quadCellInd:] = sigma1.flatten()
    meshioPy.cell_data['Sigma yy'][quadCellInd:] = sigma2.flatten()
    meshioPy.cell_data['Von Mises'][quadCellInd:] = vm.flatten()
    meshioPy.cell_data['Shear Stress'][quadCellInd:] = shear.flatten()
    meshioPy.cell_data['Maximum Principle Shear Stress'][quadCellInd:] = shear1.flatten()

    return meshioPy, deformedMesh

def vtkEigenAnalysis(K_bc, nmodes, mesh):
    """Performs eigen analysis and prepares VTK output."""
    eigVal, eigVec = sp.sparse.linalg.eigsh(K_bc, k=nmodes, which='SA')
    for ii in range(nmodes):
        nm = "eVec%d" % (ii + 1)
        # The eigenvector data has two components per node (2D solver)
        mesh.point_data[nm] = np.c_[eigVec[::2, ii], eigVec[1::2, ii]]
    return mesh