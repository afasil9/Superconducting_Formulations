# Superconducting Formulations

Numerical modelling of bulk superconductors using the **H-formulation** of Maxwell's equations.

## Overview

This repository contains models for solving electromagnetic problems in bulk superconductors. The H-formulation uses the magnetic field **H** as the unknown, which makes it well suited to the highly nonlinear behaviour of superconducting materials.

The governing equation comes from Faraday's law:

$$\mu_0 \frac{\partial \mathbf{H}}{\partial t} + \nabla \times \left( \rho \, \nabla \times \mathbf{H} \right) = 0$$

where the current density is $\mathbf{J} = \nabla \times \mathbf{H}$ and $\mathbf{E} = \rho \mathbf{J}$.

The superconductor is described by the E–J power law:

$$\mathbf{E} = E_0 \left( \frac{|\mathbf{J}|}{J_c} \right)^{n} \frac{\mathbf{J}}{|\mathbf{J}|}$$

## Cases

| Case | Description | Status |
|------|-------------|--------|
| Solid cylinder | Bulk superconducting cylinder, linear parameters | Initial version |

## Requirements

- [Software / language, e.g. Python 3.x, FEniCS, COMSOL, MATLAB]
- [Any packages]

## Usage

[How to run a model, e.g. `python solid_cylinder.py`]

## Repository structure

[Short list of the main files and folders]

## References

- [Key papers on the H-formulation you're building on]

## Author

afasil9