import numpy as np
from scipy.optimize import fsolve
from thirdParty.DirichletBC import DirichletBC
from thirdParty.guassQuad import gaussQuad

def areaCalc(L1, thetaRad, phiRad, b=0.1):
    """Calculates wing area using trigonometry."""
    return 8 * L1 * L1 / np.tan(thetaRad) - 7.5 * L1 * L1 / np.tan(phiRad) + 4 * L1 * b

def shapeCalc(L1, target_area, theta, phi, initial_b=0.1):
    """Calculates right-hand length iteratively to achieve target area. I know its inefficient"""

    # Define the objective function for the root-finding algorithm
    # The function calculates the difference between the current area and the target area.
    def objective(b):
        return areaCalc(L1, theta, phi, b) - target_area

    # Use `fsolve` to find the value of b that minimizes the objective function to zero
    # `fsolve` iteratively adjusts b to make `objective(b) = 0`
    b_solution = fsolve(objective, initial_b, xtol=0.01)

    # Return the solution for b and the calculated area based on this solution
    return b_solution[0], areaCalc(L1, theta, phi, b_solution[0])

def jacobian(derivatives, nodes):
    """Calculates Jacobian matrix for coordinate transformation."""
    J = np.array([
        [np.dot(derivatives[0], nodes[:, 0]), np.dot(derivatives[0], nodes[:, 1])],
        [np.dot(derivatives[1], nodes[:, 0]), np.dot(derivatives[1], nodes[:, 1])]
    ])
    invJ = np.linalg.inv(J)
    detJ = np.linalg.det(J)
    # Check for singularity in the Jacobian (zero determinant)
    if np.abs(detJ) < 1e-12:
        raise ValueError("Jacobian determinant is zero, check element geometry.")
    return J, invJ, detJ

def strainDisp2D(SF, nodeCoor, Jacob):
    """Calculates strain-displacement matrix for 2D elements."""
    # Extract the number of nodes (rows in nodeCoor), shape functions and derivatives, and inverse jacobian
    nne = nodeCoor.shape[0]
    sf, dsfdx, dsfde = SF['sf'], SF['dndx'], SF['dnde']
    I = Jacob['invJ']

    # Compute the derivative of displacement components in the x direction
    # np.stack combines sdfdx and zeros(nne) along a new axis, then flattens into 1D for future calculations. Less intermediate arrays improves efficiency
    r1 = np.stack([dsfdx, np.zeros(nne)], axis=-1).flatten()  # Shape function derivatives w.r.t x
    r2 = np.stack([dsfde, np.zeros(nne)], axis=-1).flatten()  # Shape function derivatives w.r.t y

    # Obtain the displacement derivatives
    # Vstack is for vertical stacking
    R = I @ np.vstack((r1, r2))  # Apply the inverse Jacobian to the shape function derivatives
    dudx, dudy = R[0], R[1]  # Displacement derivatives in the x and y direction

    # Compute the derivative of displacement components in the y direction
    r1 = np.stack([np.zeros(nne), dsfdx], axis=-1).flatten()  # Derivative of shape function w.r.t. x
    r2 = np.stack([np.zeros(nne), dsfde], axis=-1).flatten()  # Derivative of shape function w.r.t. y

    # Obtain the displacement derivatives
    R = I @ np.vstack((r1, r2))
    dvdx, dvdy = R[0], R[1]

    # Construct the strain-displacement matrix by stacking the displacement derivatives
    B = np.vstack((dudx, dvdy,
                   dudy + dvdx))  # The matrix B is a 3 X nne matrix, where each row corresponds to a strain component (xx, yy, xy)
    return B