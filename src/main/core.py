import numpy as np

def setupParams():
    """
    Function to allow the user to input parameters interactively or use default values.
    Returns a dictionary of parameters.
    """
    print("Setup simulation parameters (press Enter to use default values):\n")

    # Function to get user input with type and default value
    def get_input(prompt, default, dtype):
        while True:
            try:
                user_input = input(f"{prompt} [{default}]: ")
                if not user_input.strip():  # Use default if no input
                    return default
                return dtype(user_input)
            except ValueError:
                print(f"Invalid input. Please enter a value of type {dtype.__name__}.")

    # Input parameters
    E1 = get_input("Longitudinal Young's Modulus (Pa)(E1)", 180e9, float)
    E2 = get_input("Transverse Young's Modulus (Pa)(E2)", 7.5e9, float)
    nu12 = get_input("Poisson's Ratio (nu12)", 0.3, float)
    G12 = get_input("Shear Modulus (G12)", 6e9, float)
    quadNo = get_input("Number of Quadrature Points (quadNo)", 6, int)
    filename = get_input("Filename for the mesh", "wingProblem", str)
    L1 = get_input("Length parameter for the wing (L1)", 8, float)
    theta = get_input("Angle parameter (theta)", 75, float)
    phi = get_input("Angle parameter (phi)", 75, float)
    area = get_input("Target area of the wing", 250, float)
    meshSize = get_input("Minimum mesh size", 0.3, float)
    nne = get_input("Nodes per element (4 or 8)", 8, int)
    force = get_input("Force applied (negative for downward)", -200, float)
    eigshModes = get_input("Number of eigenmodes for analysis", 6, int)
    print("\nParameters setup complete.")
    return E1, E2, nu12, G12, quadNo, L1, theta, phi, area, meshSize, nne, force, eigshModes, filename