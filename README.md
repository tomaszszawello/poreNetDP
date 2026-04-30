# poreNetDP

`poreNetDP` is a Python package for simulating dissolution and precipitation in porous media using pore-network models.

The code was developed for studying how flow, transport, dissolution, precipitation, and pore-network geometry interact during reactive transport in porous materials. The main branch contains the most versatile version of the model, while several development branches contain more specialized setups and experimental extensions.

## Status

As of April 2026, this repository is undergoing refactoring.

The `main` branch is the most general and versatile version of the code. Other branches are also useful, but should be treated as work in progress:

* `fractures` — simulations of dissolution in discrete fracture networks
* `mip` — mixing-induced precipitation
* `benchmark` — benchmark cases and testing setups
* `calcite_iron` — simulations for a specific calcite/iron experimental setup
* `exploration` — simulations for specific calcite/gypsum experimental setups
* `metamorph` — simulations of rock transformations in nature
* `diffusion` — simulations of diffusion-controlled natural rock transformations

These branches may contain useful models, examples, or parameter sets, but APIs, file structure, and numerical details may change.

## Citation

If you use this code for scientific purposes, please cite:

> Szawełło, T., Hyman, J. D., Kang, P. K., & Szymczak, P. (2024). Quantifying dissolution dynamics in porous media using a spatial flow focusing profile. *Geophysical Research Letters*, 51, e2024GL109940. [https://doi.org/10.1029/2024GL109940](https://doi.org/10.1029/2024GL109940)

## Funding

This research is funded by the National Science Centre, Poland, under Grant 2022/47/B/ST3/03395.

## Requirements

Use Python 3.10. Some type hints and annotations may not work properly with older Python versions.

Required packages:

```bash
dill json matplotlib networkx numpy scipy vtk
```

A minimal installation can be done with:

```bash
pip install dill json matplotlib networkx numpy scipy vtk
```

## Quick start

1. Clone the repository:

```bash
git clone https://github.com/tomaszszawello/poreNetDP.git
cd poreNetDP
```

2. Install the required packages:

```bash
pip install dill matplotlib networkx numpy scipy
```

3. Set simulation parameters in `config.py`.

4. Run the simulation:

```bash
python .
```

or, equivalently:

```bash
python __main__.py
```

5. Wait for the results to appear in the simulation directory.

Initialization may take a while. For example, a network of size around `100 x 100` may take about one minute to initialize. Simulations should usually work up to network size around `200 x 200` on any PC; above that, usage of work stations is more recommended.

## Basic workflow

The typical workflow is:

1. edit parameters in `config.py`
2. build or load a network
3. solve pressure and flow
4. solve transport, dissolution, precipitation, or diffusion depending on the selected setup
5. save and plot the resulting network and simulation data

The exact workflow depends on the selected configuration and branch.

## Repository structure

The main branch contains modules for network construction, incidence matrices, pressure and transport solves, reactive processes, saving, and plotting. Important files include:

* `config.py` — simulation parameters
* `__main__.py` — main simulation entry point
* `build.py` — initialization of simulation objects
* `network.py` — network construction and graph-related classes
* `incidence.py` — incidence and sparse matrix construction
* `pressure.py` — pressure and flow calculation
* `diffusion.py` — dissolution with advection—diffusion models
* `dissolution.py` — dissolution with advection models
* `precipitation.py` — precipitation models
* `growth.py` — network evolution/growth logic
* `volumes.py` — solid volume tracking
* `draw_net.py` — plotting utilities
* `save.py` — saving and loading simulations

## Notes

This is a research code. It is actively evolving, and some parts of the codebase are being reorganized. If you are using a development branch, please check the branch-specific code carefully before using it for production runs or quantitative comparisons.

For reproducibility, it is recommended to record:

* branch name
* commit hash
* configuration file
* random seed, if used
* Python version
* package versions

## Contact

Should you have any questions, contact:

`t.szawello@uw.edu.pl`
