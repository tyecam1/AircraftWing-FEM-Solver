import numpy as np
import subprocess
import pyvista as pv
from scipy.optimize import fsolve
from .utils import areaCalc, shapeCalc

def wingMesher(filename, L1=8, theta=75, phi=75, area=200, meshSize=0.5, nne=4):
    """Generates a GMSH geometry file to mesh the cross-section of a wing."""
    # Validate input angles to ensure they fall within the range 60° ≤ θ, φ ≤ 90°
    if not (60 <= theta <= 90) or not (60 <= phi <= 90):
        raise ValueError("Both theta and phi must be between 60° and 90°.")

    # Convert theta and phi from degrees to radians for trigonometric calculations
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)

    # Call a function to compute shape parameters and the actual area based on constraints
    b, realArea = shapeCalc(L1, area, theta_rad, phi_rad)

    # Define coordinates of the wing's five corner points based on geometry
    x1, y1 = 0, 0  # Origin (Point 1)
    x2, y2 = 4 * L1, -4 * L1 * (np.cos(theta_rad) / np.sin(theta_rad))  # Point 2
    x3, y3 = 4 * L1, y2 - b  # Point 3
    x4, y4 = L1, y3 + 3 * L1 / np.tan(phi_rad)  # Point 4
    x5, y5 = 0, y4  # Point 5

    # Open a file to write the geometry in GMSH's .geo format
    with open(f"{filename}.geo", "w") as geo_file:
        # Write the five points with their mesh sizes
        geo_file.write(f"Point(1) = {{{x1}, {y1}, 0, {meshSize}}};\n")  # Fine mesh for point 1
        geo_file.write(f"Point(2) = {{{x2}, {y2}, 0, {meshSize * 2}}};\n")  # Medium mesh for point 2
        geo_file.write(f"Point(3) = {{{x3}, {y3}, 0, {meshSize}}};\n")
        geo_file.write(f"Point(4) = {{{x4}, {y4}, 0, {meshSize}}};\n")
        geo_file.write(f"Point(5) = {{{x5}, {y5}, 0, {meshSize}}};\n")

        # Line definitions // outline of the wing
        geo_file.write("Line(1) = {1, 2};\n")  # Line from Point 1 to Point 2
        geo_file.write("Line(2) = {2, 3};\n")  # Line from Point 2 to Point 3
        geo_file.write("Line(3) = {3, 4};\n")  # Line from Point 3 to Point 4
        geo_file.write("Line(4) = {4, 5};\n")  # Line from Point 4 to Point 5
        geo_file.write("Line(5) = {5, 1};\n")  # Line from Point 5 to Point 1

        # Configure mesh size along the defined lines
        geo_file.write(f"MeshSize {meshSize * 3} {{ Line{{1, 2, 3, 4, 5}} }};\n")  # Medium mesh size

        # Create a closed loop of the lines and define a surface for the wing
        geo_file.write("Curve Loop(1) = {1, 2, 3, 4, 5};\n")  # Closed loop of the lines
        geo_file.write("Plane Surface(1) = {1};\n")  # Surface enclosed by the loop

        # Configure global mesh size scaling
        geo_file.write(f"Mesh.ElementSizeFactor = {meshSize * 3};\n")  # Coarser mesh globally
        geo_file.write("Recombine Surface{1};\n")  # Use quadrilateral elements for the surface

        if nne == 8:  # Ensure second-order elements for midside nodes
            geo_file.write("Mesh.SecondOrder = 1;\n")  # Add midside nodes to edges
            # Ensure GMSH generates only 8-noded elements
            geo_file.write("Mesh.ElementOrder = 2;\n")  # Second-order elements
            geo_file.write("Mesh.SecondOrderIncomplete = 1;\n")  # Remove central nodes
            geo_file.write("Mesh.RecombineAll = 1;\n")  # Recombine globally

    # Run GMSH to generate the mesh from the .geo file
    subprocess.run([
        "gmsh",
        f"{filename}.geo",  # Input geometry file
        "-2",  # Generate 2D mesh
        "-o", f"{filename}.msh",  # Output mesh file
        "-format", "msh2",  # Use MSH2 format
        "-clscale", str(meshSize * 3),  # Scale factor for global mesh size
    ], timeout=60)

def prepMesh(filename, L1, theta, phi, area, meshSize, nne, quadNo):
    """Prepare the mesh for finite element analysis (FEA)."""
    # Generate the mesh
    wingMesher(filename, L1, theta, phi, area, meshSize, nne)
    mesh = pv.read_meshio(filename + '.msh')  # Load generated mesh
    # Set up Gaussian quadrature points and weights
    qpt, qwt = gaussQuad(quadNo)
    if nne == 4:
        con_mat = mesh.cells_dict.get(9, None)  # 4 Noded quadrilateral elements
        con_mat = con_mat[:, np.array([0, 3, 2, 1])]
    elif nne == 8:
        con_mat = mesh.cells_dict.get(23, None)
        con_mat = con_mat[:, np.array([0, 7, 3, 6, 2, 5, 1, 4])]

    nelem = len(con_mat)
    nodeCoor = mesh.points  # Node coordinate matrix
    nnodes = mesh.points.shape[0]  # Total number of nodes
    dofpn = 2  # Degrees of freedom per node (2D problem)
    dofpe = nne * dofpn  # Degrees of freedom per element
    ndof = nnodes * dofpn  # Total degrees of freedom in the mesh

    # Return all relevant data
    return mesh, con_mat, nodeCoor, nelem, ndof, qpt, qwt, dofpe, nnodes