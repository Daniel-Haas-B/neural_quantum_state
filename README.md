# Neural Quantum States: Using RBMs and NNs to simualate quantum mechanical systems
The code structure is based on this [template](https://github.com/mortele/variational-monte-carlo-fys4411) (thank you).

In this project we have tried to simulate a bosonic system of two bosons in a magnetic trap, and find the lowest energy state for this system. To do this, we have used tools such as Variational Monte Carlo, Restricted Bolzmann Machines and Feed-forward neural networks, making use also of the Blocking Method to generate the statistics. This file will show how to navigate our repository and use the programs. The main calculations are written in ``C++14``, with analysis, plotting and statistical analysis performed using ``Python3``.

## ``C++`` programs, compilation and usage
To run and compile the ``C++``, a version of [openMP](https://www.openmp.org/) is required. The compilation and linking is performed using [CMake](https://cmake.org/). A handy script ``compile_project`` is provided for easier compilation. Run

```bash
./compile_project
```
Or perform manually, using 
```bash
# Create build-directory
mkdir build

# Move into the build-directory
cd build

# Run CMake to create a Makefile
cmake ../

# Make the Makefile using two threads
make -j2
```
The linking with ``openMP`` might differ based on your system. Currently, it is configured to compile for Mac (M1 chip). Configure the ``CMakeLists.txt`` for your desired system if problems are encountered.

After compilation, five executables should be presented in the ``build`` directory. With some overlapping functionality, they briefly do:

- ``vmc``: Perform a single non-interactive vmc calculation.
- ``timing``: Perform timing on the non-interactive vmc calculation, using either analytical or numerical double derivatives.
- ``gradient``: Performs gradient decent on either a non-interactive or interactive system. .
- ``interact``: Perform a single interactive vmc calculation.
- ``parallelinteract``: Perform multiple interactive vmc calculations in parallel.

Usage can be seen by running the programs with no arguments.
