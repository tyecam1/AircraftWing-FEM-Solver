import numpy as np

def gaussQuad(npts):
    """
    Calculates the Gauss Quadrature points & weights.

    Inputs  :
    npts    : number of quadrature integration points (valid values: 2, 3, 4, 5, 6)

    Outputs :
    points  : vector quadrature points (npts)
    weights : vector quadrature weights (npts)
    """
    # Define dictionary for Gauss points and weights for known npts values
    gauss_data = {
        2: {
            "points": np.array([1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)]),
            "weights": np.array([1.0, 1.0])
        },
        3: {
            "points": np.array([0.774596669241483, 0.0, -0.774596669241483]),
            "weights": np.array([0.555555555555556, 0.888888888888889, 0.555555555555556])
        },
        4: {
            "points": np.array([0.861136311590453, 0.339981043583856, -0.339981043583856, -0.861136311590453]),
            "weights": np.array([0.347854845137454, 0.652145154862526, 0.652145154862526, 0.347854845137454])
        },
        5: {
            "points": np.array([0.0, 1.0 / 3.0 * np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0)),
                                -1.0 / 3.0 * np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0)),
                                1.0 / 3.0 * np.sqrt(5.0 + 2.0 * np.sqrt(10.0 / 7.0)),
                                -1.0 / 3.0 * np.sqrt(5.0 + 2.0 * np.sqrt(10.0 / 7.0))]),
            "weights": np.array(
                [128.0 / 225.0, (322.0 + 13.0 * np.sqrt(70.0)) / 900.0, (322.0 + 13.0 * np.sqrt(70.0)) / 900.0,
                 (322.0 - 13.0 * np.sqrt(70.0)) / 900.0, (322.0 - 13.0 * np.sqrt(70.0)) / 900.0])
        },
        6: {
            "points": np.array([0.2386191861, -0.2386191861, 0.6612093865, -0.6612093865, 0.9324695142, -0.9324695142]),
            "weights": np.array([0.4679139346, 0.4679139346, 0.3607615730, 0.3607615730, 0.1713244924, 0.1713244924])
        }
    }
    # Check if npts is a valid key in gauss_data
    if npts not in gauss_data:
        raise ValueError(f"Unsupported number of points: {npts}. Supported values: 2, 3, 4, 5, 6.")

    # Return the corresponding points and weights
    points = gauss_data[npts]["points"]
    weights = gauss_data[npts]["weights"]

    return points, weights