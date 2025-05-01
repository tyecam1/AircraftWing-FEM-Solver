import numpy as np

# Ce calculation
def elasticityTensor(E1, E2, nu12, G12):
    nu21 = nu12 * E2 / E1  # Reciprocal Poisson's ratio
    coeff = 1 / (1 - nu12 * nu21)  # Denominator for plane stress
    D = coeff * np.array([
        [E1, nu21 * E2, 0],
        [nu12 * E1, E2, 0],
        [0, 0, G12 * (1 - nu12 * nu21)]
    ])
    return D