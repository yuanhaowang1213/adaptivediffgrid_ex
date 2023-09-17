#!/bin/bash

# source $(conda info --base)/etc/profile.d/conda.sh

# conda create -y -n hyperoct python=3.8
# conda activate hyperoct


conda install -y -c anaconda libgcc-ng=9
conda install -y -c conda-forge libstdcxx-ng=9 libgomp=9 coin-or-cbc glog gflags protobuf=3.11.4 xorg-libxrandr xorg-libxinerama xorg-libxcursor xorg-libxi nibabel imageio matplotlib mrc
conda install -y cudnn=8.2.1.32 cudatoolkit-dev=11.2 cudatoolkit=11.2 -c nvidia -c conda-forge
conda install -y astunparse numpy ninja pyyaml mkl mkl-include cmake=3.19.6 cffi typing_extensions future six requests dataclasses setuptools tensorboard configargparse
conda install -y magma-cuda110 -c pytorch
pip install scipy




pip install setuptools==59.5.0
conda install -y -c dlr-sc freeimageplus

git submodule update --init --recursive --jobs 0

mkdir External
git clone git@github.com:google/or-tools.git
cd or-tools
git checkout -b 86d4c543f7
git submodule update --init --recursive --jobs 0
cd ..
git clone git@github.com:pytorch/pytorch.git
git clone git@github.com:darglein/saiga.git
cd saiga
git checkout -b a9c60bd6
git submodule update --init --recursive --jobs 0
cd ..
git clone git@github.com:RustingSword/tensorboard_logger.git
cd tensorboard_logger
git checkout -b 22dc162
git submodule update --init --recursive --jobs 0
cd ..

# install pytorch
cd External/pytorch/
git checkout v1.9.1
# git checkout v1.12.1
git submodule update --init --recursive --jobs 0

# conda install -c conda-forge gcc=9
if command -v g++-11 &> /dev/null
then
    export CC=$(which gcc-11)
    export CXX=$(which g++-11)
    export CUDAHOSTCXX=$(which g++-11)
    echo "Using g++-9"
elif command -v g++-9 &> /dev/null
then
    export CC=$(which gcc-9)
    export CXX=$(which g++-9)
    export CUDAHOSTCXX=$(which g++-9)
    echo "Using g++-9"
elif command -v g++-7 &> /dev/null
then
    export CC=$(which gcc-7)
    export CXX=$(which g++-7)
    export CUDAHOSTCXX=$(which g++-7)
    echo "Using g++-7"
else
    echo "No suitable compiler found. Install g++-7 or g++-9"
    exit
fi

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CONDA=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

python setup.py install

cd ../..

cd External/or-tools/
git apply ../../patches/or_patch.patch
cd ../..




mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="${CONDA}/lib/python3.8/site-packages/torch/;${CONDA}" ..
make -j64
