import numpy as np
import multiprocessing
import csv
from scipy.optimize import differential_evolution
from .materials import elasticityTensor
from .mesh import prepMesh
from .solver import (elementMatrices, assembleGlobalSys,
                    boundaryConditions, stress)

def processMaterialCombination(material):
    """Optimizes geometric parameters for given material properties."""
    # Unpack material properties
    E1, E2, G12 = material

    print(f"Optimising for E1 = {E1}, E2 = {E2}, G12 = {G12}...")

    # Precompute elasticity tensor
    Ce = elasticityTensor(E1, E2, 0.3, G12)  # nu12 is hardcoded as 0.3 here

    # Objective function to minimise
    def objective(params):  # Design variables to optimise
        theta, phi = params
        try:
            print(f"Testing theta = {theta}, phi = {phi}...")
            # Mesh generation and preprocessing
            filename = f"wingProblem_E1_{E1 / 100000000}_E2_{E2 / 100000000}_G12_{G12 / 100000000}_Theta{round(theta, 4)}_Phi{round(phi, 4)}"

            meshioPy, con_mat, nodeCoor, nelem, ndof, qpt, qwt, dofpe, nnodes = prepMesh(
                filename, 8, theta, phi, 250, 0.4, 8, 6
            )

            # Elemental matrices and assembly
            K, kEl, aEl, BEl, FwEl = elementMatrices(
                nelem, dofpe, ndof, nodeCoor, con_mat, qpt, qwt, Ce, 8
            )
            K = assembleGlobalSys(K, kEl, con_mat)

            # Apply boundary conditions
            K_bc, u, ux, uy = boundaryConditions(nodeCoor, K, ndof, -150, FwEl, con_mat)

            # Stress calculations
            sigma1, sigma2, shear, vm, shear1 = stress(u, con_mat, nelem, BEl, Ce)

            # Maximum values
            max_shear = np.max(shear)
            max_vm = np.max(vm)

            # Weighted metric
            weighted_metric = (
                    0.75 * max_vm + 0.25 * max_shear
            )
            # Handle non-finite results
            if not np.isfinite(weighted_metric):
                raise ValueError(f"Non finite result, weighted_metric = {weighted_metric}")
            return weighted_metric
        # Catch and handle errors during optimisation
        except Exception as e:
            print(f"Error: {e}")
            # Return a high value to indicate failure
            return 1e12

def optimiseStress():
    """Optimizes stress distribution for different material properties."""
    # Material property ranges
    E1_range = np.arange(130e9, 230e9 + 1, 25e9)
    E2_range = np.arange(5e9, 10e9 + 0.1, 2.5e9)
    G12_range = np.arange(4e9, 8e9 + 0.1, 2e9)

    # Prepare material property combinations
    E1_vals, E2_vals, G12_vals = np.meshgrid(E1_range, E2_range, G12_range)
    # Stack the combinations into a 2D array
    material_combinations = np.vstack([E1_vals.ravel(), E2_vals.ravel(), G12_vals.ravel()]).T

    # File to save results
    results_file = "optimisation_results.csv"

    # Use multiprocessing for parallel optimization
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Map the processMaterialCombination function to each material combination
        results = pool.map(processMaterialCombination, material_combinations)

    # Write results to CSV file
    with open(results_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "E1", "E2", "G12", "theta", "phi",
            "max_sigma1", "max_sigma2", "max_shear", "max_vm", "max_shear1",
            "ux", "uy"
        ])
        writer.writerows(results)

    print(f"Optimisation complete. Results saved to {results_file}.")

def measureStresses():
    """Calculates stresses for various angle combinations."""
    # Create or open a CSV file to store results
    with open('completeResultsForOptimalMaterial.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(['Theta', 'Phi', 'Sigma1', 'Sigma2', 'Shear', 'VM', 'Shear1', 'Error'])

        # Define constants
        E1 = 2.05e11
        E2 = 1e10
        G12 = 4e9
        nu12 = 0.3
        quadNo = 2
        L1 = 8
        area = 240
        meshSize = 0.5
        nne = 4
        force = -250
        filename = 'completeResultsForOptimalMaterial'

        # Loop through theta and phi values from 60 to 90 in steps of 2
        for theta in range(60, 91, 2):
            for phi in range(60, 91, 2):
                try:
                    # Calculate material elasticity tensor
                    Ce = elasticityTensor(E1, E2, nu12, G12)

                    # Mesh generation and preprocessing
                    meshioPy, con_mat, nodeCoor, nelem, ndof, qpt, qwt, dofpe, nnodes = prepMesh(
                        filename, L1, theta, phi, area, meshSize, nne, quadNo
                    )

                    # Initializing & populating elemental arrays
                    K, kEl, aEl, BEl, FwEl = elementMatrices(nelem, dofpe, ndof, nodeCoor, con_mat, qpt, qwt, Ce, nne)

                    # Assemble global stiffness matrix
                    K = assembleGlobalSys(K, kEl, con_mat)

                    # Apply boundary conditions and calculate displacements
                    K_bc, u, ux, uy = boundaryConditions(nodeCoor, K, ndof, force, FwEl, con_mat)

                    # Calculate stresses
                    sigma1, sigma2, shear, vm, shear1 = stress(u, con_mat, nelem, BEl, Ce)

                    # Write results to the CSV file
                    writer.writerow(
                        [theta, phi, np.max(sigma1), np.max(sigma2), np.max(shear), np.max(vm), np.max(shear1), ''])

                except Exception as e:
                    # Handle errors and skip the iteration
                    print(f"Error processing theta={theta}, phi={phi}: {e}")
                    writer.writerow([theta, phi, 'Error', 'Error', 'Error', 'Error', 'Error', str(e)])

    print("Simulations complete. Results saved to 'completeResultsForOptimalMaterial.csv'")