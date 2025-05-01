
# Aircraft Wing Finite Element Method Solver

A finite element analysis tool for aircraft wing structural optimisation with material property and geometric parameter optimisation.

## Features
- Parametric 2D wing generation (birdseye)
- Orthotropic material modeling (composites). Capable of 8 noded element calculations.
- Stress minimisation using differential evolution
- Multi-core parallel processing
- ParaView-compatible VTK output, including deformed model.
- Sparsity pattern visualisation

## Installation

```bash
# Clone repository
git clone https://github.com/tyecam1/AircraftWing-FEM-Solver.git
cd AircraftWing-FEM-Solver

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# OR
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Install GMSH (required for meshing)
# Download from https://gmsh.info/ and add to PATH
Basic Usage
python
from wingfea import setupParams, prepMesh
from wingfea.solver import run_analysis

# Interactive parameter setup
params = setupParams()

# Run full analysis pipeline
results = run_analysis(*params)

# Save results
results['deformed_mesh'].save('wing_deformed.vtk')
Optimisation
python
from wingfea.optimisation import optimiseStress

# Run material optimisation study
optimiseStress()  # Results saved to optimisation_results.csv

#Project Structure
wing-fea-optimiser/
├── src/
│   └── wingfea/
│       ├── __init__.py
│       ├── core.py          # Parameter setup
│       ├── materials.py     # Material models
│       ├── mesh.py          # GMSH integration
│       ├── solver.py        # FEA core
│       ├── optimisation.py  # DE optimisation
│       ├── utils.py         # Helper functions
│       └── visualisation.py # VTK/ParaView tools
├── tests/                   # Unit tests
├── examples/                # Usage examples
├── requirements.txt         # Python dependencies
├── setup.py                 # Package config
└── LICENSE
```

## Dependencies
Python 3.8+

NumPy

SciPy

PyVista

Matplotlib

GMSH (system install)

## License
MIT License - See LICENSE for details.

Optimisation code developed using Cardiff University MEng finite element method module content.
