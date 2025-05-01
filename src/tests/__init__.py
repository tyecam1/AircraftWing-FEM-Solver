"""Finite Element Analysis package for wing structure optimization."""
from .core import setupParams
from .mesh import wingMesher, prepMesh
from .materials import elasticityTensor
from .solver import (jacobian, strainDisp2D, quadCalcs, assembleGlobalSys,
                    elementMatrices, boundaryConditions, stress)
from .optimization import processMaterialCombination, optimiseStress, measureStresses
from .visualization import plotSparsities, setupParaviewData, vtkEigenAnalysis

__version__ = "0.1.0"