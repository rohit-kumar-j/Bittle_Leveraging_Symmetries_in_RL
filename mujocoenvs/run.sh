#!/bin/sh
set -xe # build output

#curr_dir=$(pwd)
# dyld_insert_libraries=/home/rohit/github/dlar/pybindtest/include/mujoco/lib/libmujoco.so python3 test.py
export LD_LIBRARY_PATH="$(pwd)""/external/mujoco/lib64"
python3 test.py
