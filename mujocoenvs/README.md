# Prerequesits 
build and install mujoco using cmake. Use the `prereq.sh` script for this:

```bash
chmod +x prereq.sh
./prereq.sh
```

After running this, you should find mujoco like this:

```bash
+--mujocoenvs
    |
    +--external/    # Installation location of thirdparty libs
        |--mujoco 
        |--pybind11 
    |
    +--thirdparty/ # Src directories of thirdparty repositories
        |--mujoco 
        |--pybind11
```

Install the python development package if not installed: 
(This is used for include headers and libraries which are detected during runtime)

```bash
sudo apt-get install python-dev  # Ubuntu
sudo dnf install python-devel    # Fedora
```

## Building and binding external libraies with pybind11

```bash
chmod +x build.sh
./build.sh
```

## Running the script
```bash
chmod +x run.sh
./run.sh
```


## Clean install of all dependencies
```bash
chmod +x clean.sh
./clean.sh
```

