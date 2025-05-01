from setuptools import setup, find_packages

setup(
    name="wingfea",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pyvista>=0.32.0",
        "matplotlib>=3.4.0",
        "gmsh>=4.8.0",
    ],
    python_requires=">=3.8",
)