# ZigZag-IMC
This repository presents the extended version of ZigZag, a HW Architecture-Mapping Design Space Exploration (DSE) Framework.
This extension is specifically designed to support In-Memory Computing (IMC).
ZigZag-IMC bridges the gap between algorithmic DL decisions and their IMC acceleration cost on specialized accelerators through a fast and accurate HW cost estimation. 

## Prerequisite

To get started, you can install all packages directly through pip using the pip-requirements.txt with the command:

`$ pip install -r requirements.txt`

## Getting Started
The main script is main_onnx.py, which takes into 3 input files:
- .onnx: workload definition.
- mapping: user-provided spatial mapping or spatial mapping restriction.
- accelerator: hardware definition.

The repository includes three examples provided in the `run.sh` script, which are:
- an example for a pure digital PE-based hardware template.
- an example for an SRAM-based Digital In-Memory Computing hardware template.
- an example for an SRAM-based Analog In-Memory Computing hardware template.

The output will be saved under `outputs/` folder.

## New features
In this novel version, in addition to the features found in the basic zigzag, we have introduced several new capabilities:
- **New cost models**: Added support for SRAM-based Analog In-Memory Computing and Digital In-Memory Computing (28nm).
- **Dataflow optimization***: Inner layer data will remain in lower memory if it is expected to be used by the next layer along the same branch.
- **Mix Spatial Mapping***: Mix user-defined spatial mapping is now allowed (e.g. `inputs/examples/mapping/default_imc.py`).
- **Diagonal OX/OY Mapping***: Computing array now supports diagonal OX/OY mapping.
- **Automatic Mix Spatial Mapping***: Spatial mapping will be autogenerated if a `spatial_mapping_hint` is provided in the mapping file.
- **Simulation Speedup**: Only the three spatial mappings with the highest hardware utilization will be assessed to ensure fast simulation speed.

*: features are planned to be integrated into the base ZigZag framework.

## Publication pointers
- J. Sun, P. Houshmand and M. Verhelst, "Analog or Digital In-Memory Computing? Benchmarking through Quantitative Modeling," Proceedings of the IEEE/ACM Internatoinal Conference On Computer Aided Design (ICCAD), October 2023.

- P. Houshmand, J. Sun and M. Verhelst, "Benchmarking and modeling of analog and digital SRAM in-memory computing architectures," arXiv preprint arXiv:2305.18335 (2023).


