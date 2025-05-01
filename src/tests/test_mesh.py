import pytest
from wingfea.mesh import wingMesher
import os

def test_wingMesher(tmp_path):
    filename = str(tmp_path / "test_mesh")
    wingMesher(filename, L1=8, theta=75, phi=75, area=200, meshSize=0.5, nne=4)
    assert os.path.exists(filename + ".geo")
    assert os.path.exists(filename + ".msh")