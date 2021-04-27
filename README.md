# CERTIFICATION SUITE FOR MACHINE LEARNING CARE LABELS

This is a prototype certification suite, which generates **Care Labels** for trustworthy and resource-aware machine learning.
It is closely related to research work which is currently reviewed for ECML 2021, where we explain the concepts of our certification suite in detail.
**This is only a blinded preview for paper reviewers**, upon acceptance we will transfer this repository to the authors' GitHub.

## Results
You can find all results (logs & care labels) in the `results` directory.

## Docker

Since various dependencies (software and data) are required, we provide a docker image for easier usage. 

Build the docker image:

```bash
docker build -t mlcl .
```

Start the container:

```bash
docker run -it --rm --name mlcl -v `pwd`:/src/mlcl/ mlcl bash
```

Inside the container run:

### MRF Experiments

```bash
# Set /src/mlcl/ as workdir.
cd /src/mlcl/

# CPU
python3 main.py --implementation PXMRF --config cfg/mrf_bp.json --repeats 5 --log-dir results/mrflogs/ --benchmark grid_nodes14_states2_nsamples50000.pkl --scaling-data /usr/local/data/ --out label_mrf_bp_cpu.svg
python3 main.py --implementation PXMRF --config cfg/mrf_jt.json --repeats 5 --log-dir results/mrflogs/ --benchmark grid_nodes14_states2_nsamples50000.pkl --scaling-data /usr/local/data/ --out label_mrf_jt_cpu.svg

# GPU
export PX_EXTINF=/usr/local/lib/python3.8/dist-packages/pxpy/lib/libpx_ext_culbp.so
python3 main.py --implementation PXMRF --config cfg/mrf_bp_gpu.json --repeats 5 --log-dir results/mrflogs/ --benchmark grid_nodes14_states2_nsamples50000.pkl --scaling-data /usr/local/data/ --out label_mrf_bp_gpu.svg

export PX_EXTINF=/usr/local/lib/libexternal_gpu_single_buffer.so
python3 main.py --implementation PXMRF --config cfg/mrf_jt_gpu.json --repeats 5 --log-dir results/mrflogs/ --benchmark grid_nodes14_states2_nsamples50000.pkl --scaling-data /usr/local/data/ --out label_mrf_jt_gpu.svg

```

### DNN Experiments

```bash
# Set /src/mlcl/ as workdir.
cd /src/mlcl/

# To generate care labels based on our logs, run:
python3 generate_dnn_labels.py

# To run DNN experiments, you have to install ImageNet locally (explained in more detail below) and mount it into the container
# Those experiments (especially the robustness tests) might take several days or even weeks, even with state-of-the-art hardware
python3 main.py --benchmark [imagenet_directory] --config cfg/alexnet_cpu.json --log-dir results/dnn-results/ 
python3 main.py --benchmark [imagenet_directory] --config cfg/alexnet_gpu.json --log-dir results/dnn-results/ 

python3 main.py --benchmark [imagenet_directory] --config cfg/mobilenet_cpu.json --log-dir results/dnn-results/ 
python3 main.py --benchmark [imagenet_directory] --config cfg/mobilenet_gpu.json --log-dir results/dnn-results/ 

python3 main.py --benchmark [imagenet_directory] --config cfg/resnet_cpu.json --log-dir results/dnn-results/ 
python3 main.py --benchmark [imagenet_directory] --config cfg/resnet_gpu.json --log-dir results/dnn-results/ 

python3 main.py --benchmark [imagenet_directory] --config cfg/vgg_cpu.json --log-dir results/dnn-results/ 
python3 main.py --benchmark [imagenet_directory] --config cfg/vgg_gpu.json --log-dir results/dnn-results/ 
```

If you don't want to use Docker, you can install and run all functionality on your own system.

## Requirements
We tested our software on `Ubuntu 20.04`, with Intel CPUs and NVIDIA GPUs, other architures are unlikely to be supported.

## Setup
Use python3.8 or create an Anaconda environment, and install required packages
```bash
pip3 install -r requirements.txt
```

We use Intel's "Running Average Power Limit" (RAPL) for measuring the energy consumption.
Install it from the corresponding repository:

```bash
git clone https://github.com/wkatsak/py-rapl
cd py-rapl
pip3 install .
```

For using it, you might have to change some access permissions, exemplary commands can be found in `set_rapl_rights.sh`.

### DNN Specifics

Per default, PyTorch will be installed for GPU use, but you can simply change the according lines in the `requirements.txt`.

If you want to perform experiments on your own, download the `ImageNet` data as well as robustness data from <https://github.com/hendrycks/robustness>. Make sure that all three lie in the same directory, e.g. `<path>/imagenet/...`, `<path>/imagenet-c/...`, `<path>/imagenet-p/...`.
For the latter, we provide a small helper script `download_imagenet_robustness.sh`.

For Noise robustness tests we use `cleverhans` attacks. The current version is not yet on PyPI, so install it via

```bash
git clone https://github.com/cleverhans-lab/cleverhans
cd cleverhans
pip3 install .
```

### MRF Specifics

The PyPI version of `pxpy` has some bugs that currently prevent GPU usage.
They can be manually fixed by modifying the installed `__init__.py` file in the `pxpy` directory:
- replace line 797  (`EXTINF = ext_lib.external(itype,vtype)`) with `EXTINF = ext_lib.external(self.itype,self.vtype)`
- replace line 873  (`EXTINF = ext_lib.external(itype,vtype)`) with `EXTINF = ext_lib.external(self.itype,self.vtype)`
- replace line 2251 (`EXTINF = ext_lib.external(itype,vtype)`) with `EXTINF = ext_lib.external(switch_type(itype), switch_type(vtype))`

You can download our synthetic data from <https://www.dropbox.com/sh/bcawvws67uy0v9s/AADF2TP6SVDEVeUahSUidvwVa?dl=0>, the directory with the data (`.pkl` files) has to be passed to the software via `--scaling-data`.

## Software Usage
The verification suite can be run via the `main.py`, it creates a whole bunch of logfiles, and finally aggregates them into the care label.

#### DNN Experiments
During our paper experiments we worked with multiple prototype scripts, which together generated a big `.csv` report, that was then assessed for generating the care labels.
You can inspect this report (`results` folder) and generate the care labels via running the `generate_dnn_labels.py` script.
Note that we intend to change the reported `mCE` to `relative mCE` for the Corruption robustness test in the camera-ready submission, so values and ranking can slightly change.

We incorporated all DNN functionality into our main software, but do not yet have the complete logs and results for sharing.
You can however already perform the experiments yourself by running `python main.py` (pass the directory with `ImageNet` for `--benchmark`).
You can change defaults by passing different configurations (located in `cfg`).
Note that performing robustness tests for DNN models takes several days even on modern hardware like `Nvidia A100` GPUs.

#### MRF Experiments (CPU)
Experiments are run by passing `PXMRF` as implementation, and corresponding MRF configs.
Pass the directory with the `.pkl` files as `--scaling-data`, and any of the file names for `--benchmark` (we used `grid_nodes14_states2_nsamples50000.pkl` in our experiments).

Be aware that running experiments might take some time, especially with JT und increasing number of `repeats`.
You can thus make use of our experiment logs by passing the folder `results/mrflogs` as `--log-dir`, so experiments are not rerun from scratch.
The resulting commands are

BP on CPU:
`python main.py --implementation PXMRF --config cfg/mrf_bp.json --repeats 5 --log-dir results/mrflogs/ --benchmark grid_nodes14_states2_nsamples50000.pkl --scaling-data [data-dir] --out label_mrf_bp_cpu.svg`

JT on CPU:
`python main.py --implementation PXMRF --config cfg/mrf_jt.json --repeats 5 --log-dir results/mrflogs/ --benchmark grid_nodes14_states2_nsamples50000.pkl --scaling-data [data-dir] --out label_mrf_jt_cpu.svg`

If you want to rerun and generate new logs, firstly `export OMP_NUM_THREADS=1` for deactivating the `pxpy` internal CPU parallelization, and then pass an empty `--log-dir`.


#### MRF Experiments (GPU)
For running `pxpy` on GPUs, you first have to set the `PX_EXTINF` variable towards the separately compiled inference engines.
For LBP inference, simply `export PX_EXTINF=[pxpy install directory]/lib/libpx_ext_culbp.so`, and run the software with `--config cfg/mrf_bp_gpu.json`.


For JT inference, we compiled a custom shared object which can be found in the `lib` directory of this repository, so `export PX_EXTINF=lib/libexternal_gpu_single_buffer.so`, and use `--config cfg/mrf_jt_gpu.json`.
Associated logs from our experiments are also shared in the `results/mrflogs/` directory.
