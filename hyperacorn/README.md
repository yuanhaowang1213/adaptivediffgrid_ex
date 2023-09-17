### Install Instructions

* Prepare Host System (Ubuntu)
```shell
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt install g++-9
g++-9 --version # Should Print Version 9.4.0 or higher
```
* Create Conda Environment

```shell
conda create -y -n HyperAcorn python=3.8.1
conda activate HyperAcorn

conda install -y -c conda-forge coin-or-cbc glog gflags protobuf=3.11.4
conda install -y cudnn=8.2.1.32 cudatoolkit-dev=11.2 cudatoolkit=11.2 -c nvidia -c conda-forge
conda install -y astunparse numpy ninja pyyaml mkl mkl-include cmake=3.19.6 cffi typing_extensions future six requests dataclasses setuptools tensorboard configargparse
conda install -y magma-cuda110 -c pytorch

#conda install -c conda-forge xorg-libx11
#pip install setuptools==59.5.0
 conda install -c dlr-sc freeimageplus 
```

* Compile Pytorch (Don't use the conda/pip2 package!)

 ```shell
conda activate HyperAcorn
git clone git@github.com:pytorch/pytorch.git
cd pytorch
git checkout v1.9.1
git submodule update --init --recursive --jobs 0

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CC=gcc-7
export CXX=g++-7
export CUDAHOSTCXX=g++-7
python setup.py install
 ```

* Compile HyperAcorn

```shell
conda activate HyperAcorn
git clone git@github.com:darglein/HyperAcorn.git
cd HyperAcorn
git submodule update --init --recursive --jobs 0

cd External/or-tools/
git apply ../../patches/or_patch.patch
cd ../..

export CONDA=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CC=gcc-7
export CXX=g++-7
export CUDAHOSTCXX=g++-7

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="${CONDA}/lib/python3.8/site-packages/torch/;${CONDA}" ..
make -j10


conda install -c conda-forge xorg-libxrandr xorg-libxinerama xorg-libxcursor xorg-libxi

* Prepare Host System (Ibex)


conda create -y -n HyperAcorn python=3.8
conda activate HyperAcorn

# conda install -y cudatoolkit-dev=11.2.2 -c conda-forge

export TMPDIR=/home/wangy0k/tmp
conda install cudnn=8.2.1.32 cudatoolkit=11.2 cudatoolkit-dev=11.2.2 -c conda-forge

conda install -y cudnn cudatoolkit cudatoolkit-dev -c conda-forge

conda install -y astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake=3.19.6 cffi typing_extensions future six requests dataclasses
conda install -y magma-cuda110 -c pytorch
conda install -y -c conda-forge coin-or-cbc glog gflags protobuf=3.11.4 freeimage=3.17 tensorboard configargparse setuptools=59.5.0


cd pytorch
git checkout v1.9.1
git submodule update --init --recursive --jobs 0

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CC=/sw/csi/gcc/10.2.0/bin/gcc
export CXX=/sw/csi/gcc/10.2.0/bin/g++
export CUDAHOSTCXX=/sw/csi/gcc/10.2.0/bin/g++
#export CUDA_NVCC_EXECUTABLE=/sw/csgv/cuda/11.2.2/el7.9_binary/bin/nvcc
python setup.py install


For Ibex install gcc-10
module load gcc/10.2.0 

export Torch_DIR=/home/wangy0k/anaconda3/envs/HyperAcorn/lib/python3.8/site-packages/torch
export CC=/sw/csi/gcc/10.2.0/bin/gcc
export CXX=/sw/csi/gcc/10.2.0/bin/g++
export CUDAHOSTCXX=/sw/csi/gcc/10.2.0/bin/g++
#export CUDA_TOOLKIT_ROOT_DIR=/sw/csgv/cuda/11.2.2/el7.9_binary 
cmake -DCMAKE_PREFIX_PATH="${CONDA}/lib/python3.8/site-packages/torch/;${CONDA}" ..


new workstation

conda install -c conda-forge xorg-libxrandr xorg-libxinerama xorg-libxcursor xorg-libxi
