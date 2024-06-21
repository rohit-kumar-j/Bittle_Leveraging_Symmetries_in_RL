#!/bin/sh
set -xe # enable build output

jobs=$(nproc)
proj_dir=$(pwd)
# Check with command `git tag` to list all the tags
mj_ver=3.1.6        # or whichever latest branch
pybind11_ver=v2.12.0 # or whichever latest branch

mkdir -p thirdparty
cd thirdparty

set +xe # disable build output
mj_src_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )/mujoco
if test -d "$mj_src_path" ; then
  echo "Mujoco Src exists."
else
  echo "Mujoco Src does not exist... cloning"
  git clone https://github.com/google-deepmind/mujoco.git --recurse
fi
set -xe # enable build output

cd mujoco
git checkout "$mj_ver"

# Building with cmake
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../../external/mujoco
cmake --build build --target all --parallel "$jobs"
cmake --install build

# cd "$proj_dir"
# cd thirdparty
set +xe # disable build output
pybind11_src_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )/pybind11
if test -d "$pybind11_src_path" ; then
  echo "Pybind11 Src exists."
else
  echo "Pybind11 Src does not exist... cloning"
  git clone https://github.com/pybind/pybind11.git --recurse
fi
set -xe # enable build output

cd pybind11
git checkout "$pybind11_ver"

# Building with cmake
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../../external/pybind11
cmake --build build --target all --parallel "$jobs"
cmake --install build
