# Neural Quantum States: Using RBMs and NNs to simualate quantum mechanical systems
The code structure is based on this [template](https://github.com/mortele/variational-monte-carlo-fys4411) (thank you).

In this project we have tried to simulate a bosonic system of two bosons in a harmonic oscillator trap, and find the lowest energy state for this system. To do this, we have used tools such as Variational Monte Carlo and Restricted Bolzmann Machines, making use also of the Blocking Method to generate the statistics. This file will show how to navigate our repository and use the programs. The main calculations are written in ``C++14``, with analysis, plotting and statistical analysis performed using ``Python3``.

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
Currently this project is configured to compile for Mac (M1 chip). Configure the ``CMakeLists.txt`` for your desired system if problems are encountered.

After compilation, there is a main executable, `nqs`, which performs all the simulations and should be presented in the ``build`` directory.
This exevutable accepts several arguments, but usage can be seen by running the programs with no arguments. Additionally, if no arguments are passed, the program is run with the default values. Also, these arguments are incremental, so if only the first 4 are passed, for example, all the following will be the default values. The arguments which can be fed to this main executable are

- ``numberOfDimensions``: Dimension of the quantum system
- ``numberOfParticles``: Number of bosons in the trap
- ``numberOfHiddenNodes``: Number of hidden nodes for the RBM
- ``numberOfMetropolisSteps``: (log2) Number of steps in the metropolis sampling algorithm
- ``numberOfEquilibrationSteps``: (log2) Number of equilibration steps in the metropolis sampling algorithm 
- ``importanceSampling``: Choice between Metropolis (0) or Metropolis-Hastings (1) sampling
- ``optimizerType``: Type for the optimizer. Accepted optimizers are vanillaGD, momentumGD, adamGD, rmspropGD
- ``LearningRate``: Learning rate for the optimization
- ``filename``: Name for the file with sample outputs
- ``detailed``: If true, outputs (to filename + "_detailed") all Metropolis steps sample values along the simualtion. Used in blocking algorithm
- ``beta1``: Momentum parameter related to the Adam optimizer. Used for the gridsearch of the best model
- ``beta2``: Momentum parameter related to the Adam optimizer. Used for the gridsearch of the best model
- ``eps``: Momentum parameter related to the Adam optimizer. Used for the gridsearch of the best model

Adittionally, to compare with our implementations, the srcipt `netket.py` inside of `Analysis` directory can be used to optimize the ground state of the same system with a multivariate Gaussian ansatz.