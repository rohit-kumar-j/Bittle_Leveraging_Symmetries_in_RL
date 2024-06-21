import sys
import os

# Note; The enable lazy detection of libraies durin runtime
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)

dir_path = os.path.dirname(os.path.realpath(__file__))

# adding Folder_2 to the system path
sys.path.insert(0, dir_path+"/")
sys.path.insert(0, dir_path+"/external/mujoco/lib64/")

import example  # pyright:ignore

def main():
    print("Hello World!")


if __name__ == "__main__":
    a = example.add(1, 2)
    b = example.add1(1, 2)
    c = example.add2(1, 2)
    print("\na:", a, "\nb:", b, "\nc:", c,
          "\nmeow string:", example.meow, "\n")
    print("\nmj_version:", example.my_mj_func(), "\n")
    #help(example)
