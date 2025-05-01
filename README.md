
# Wing FEA Optimizer

A finite element analysis tool for aircraft wing structural optimization with material property and geometric parameter optimization.

## Features
- Parametric wing cross-section generation
- Orthotropic material modeling (composites)
- Stress minimization using differential evolution
- Multi-core parallel processing
- ParaView-compatible VTK output
- Sparsity pattern visualization

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
Optimization
python
from wingfea.optimization import optimiseStress

# Run material optimization study
optimiseStress()  # Results saved to optimisation_results.csv

#Project Structure
wing-fea-optimizer/
├── src/
│   └── wingfea/
│       ├── __init__.py
│       ├── core.py          # Parameter setup
│       ├── materials.py     # Material models
│       ├── mesh.py          # GMSH integration
│       ├── solver.py        # FEA core
│       ├── optimization.py  # DE optimization
│       ├── utils.py         # Helper functions
│       └── visualization.py # VTK/ParaView tools
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