# CERTIFICATION SUITE FOR MACHINE LEARNING CARE LABELS

**MORE CODE WILL BE ADDED SOON** 

This is a prototype certification suite, which generates **Care Labels** for trustworthy and resource-aware machine learning.
It is closely related to research work which is currently reviewed for ECML 2021, where we explain the concepts of our certification suite in detail.
**This is only a blinded preview for paper reviewers**, upon acceptance we will transfer this repository to the authors' GitHub.

## Requirements
We tested our software on `Ubuntu 18.04`, with Intel CPUs and NVIDIA GPUs, other architures are unlikely to be supported.

## Setup

Install python3.8 (or create an Anaconda environment)

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.8
```
Install required packages
```bash
pip3 install -r requirements.txt
```

#### Data
You can download our synthetic data from <https://www.dropbox.com/sh/bcawvws67uy0v9s/AADF2TP6SVDEVeUahSUidvwVa?dl=0>, ideally extract the `.pkl` files into a new directory named `data`.

#### RAPL Rights
We use Intel's "Running Average Power Limit" (RAPL) for measuring the energy consumption.
For using it, you might have to change some access permissions, commands can be found in `set_rapl_rights.sh`.

#### GPU Support
`coming soon`

## Usage
Firstly, run `export OMP_NUM_THREADS=1` for deactivating the `pxpy` internal CPU parallelization.

#### MRF Experiments (CPU)
Run `python main.py`, the default command-line arguments should work fine but allow for some customization (e.g. for switching between the JT and BP configs).
The resulting care label content gets printed in the command line and an `.svg` graphic with the label itself will be generated.

Be aware that running experiments might take some time, especially with using JT und increasing number of `repeats`.
The high runtime stems from the different profiling measurements (runtime, memory, energy), which are all run separately.
We will soon also publish our logs as created by the software, for faster reproducibility.

#### MRF Experiments (GPU)
`coming soon`

#### DNN Experiments (CPU)
`coming soon`

#### DNN Experiments (GPU)
`coming soon`
