#!/bin/sh
set -xe # build output

curr_dir=$(pwd)
echo "$curr_dir"

clang++ -O3 -Wall -shared -std=c++11 -fPIC -I"$curr_dir"/external/mujoco/include/ -I"$curr_dir"/external/pybind11/include/ $(pkg-config --cflags --libs python) -L"$curr_dir"/external/mujoco/lib64/ -lmujoco mymodule.cpp -o example$(python3-config --extension-suffix)
