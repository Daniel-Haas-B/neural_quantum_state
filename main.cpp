#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "Hamiltonians/harmonicoscillator.h"
#include "InitialStates/initialstate.h"
#include "Math/random.h"
#include "Solvers/metropolis.h"
#include "Solvers/metropolishastings.h"
#include "WaveFunctions/gaussianbinary.h"
#include "WaveFunctions/neuralwavefunction.h"
#include "particle.h"
#include "sampler.h"
#include "system.h"
#include <omp.h>

using namespace std;

int main(int argv, char **argc)
{
    // Set default paramters

    unsigned int numberOfDimensions = 2;
    unsigned int numberOfHiddenNodes = 5;
    unsigned int numberOfParticles = 2;
    unsigned int numberOfMetropolisSteps = (unsigned int)pow(2, 16);
    unsigned int numberOfEquilibrationSteps = (unsigned int)pow(2, 14);
    double omega = 1.0;      // Oscillator frequency.
    double stepLength = 0.6; // Metropolis step length.
    double lr = 0.01;        // Learning rate for gradient descent.
    double beta1 = 0.8;      // Beta1 for Adam
    double beta2 = 0.8;      // Beta2 for Adam
    double eps = 1e-8;       // Epsilon for Adam
    // double dx = 1e-4;
    bool importanceSampling = false;
    string optimizerType = "vanillaGD";
    string rngType = "pcg";
    double D = 0.5;
    string filename = "";
    string filename_samples = "";
    string filename_posistions = "";
    bool detailed = false;
    bool interaction = false;

    // If no arguments are given, show usage.
    if (argv == 1)
    {
        cout << "Hello! Usage:" << endl;
        cout << "./vmc #dims #particles #log10(metropolis-steps) "
                "#log10(equilibriation-steps) stepLength "
                "importanceSampling? analytical? detailed? filename"
             << endl;
        cout << "#dims, int: Number of dimensions" << endl;
        cout << "#particles, int: Number of particles" << endl;
        cout << "#log2(metropolis steps), int/double: log2 of number of steps, i.e. 6 gives 2^6 steps" << endl;
        cout << "#log2(@-steps), int/double: log2 of number of equilibriation steps, i.e. 6 gives 2^6 steps" << endl;
        cout << "stepLenght, double: How far should I move a particle at each MC cycle?" << endl;
        cout << "Importantce sampling?, bool: If the Metropolis Hasting algorithm is used. Then stepLength serves as Delta t" << endl;
        cout << "optimizerType, string: What optimizer to use. Options are vanillaGD, momentumGD, adamGD, rmspropGD" << endl;
        cout << "interaction?, bool: If the interacting gaussian should be used. Defaults to false" << endl;
        cout << "filename, string: If the results should be dumped to a file, give the file name. If none is given, a simple print is performed." << endl;
        cout << "detailed?, bool: If the results should be printed in detail. Defaults to false" << endl;
        // return 0;
    }

    if (argv >= 2)
        numberOfDimensions = (unsigned int)atoi(argc[1]);
    if (argv >= 3)
        numberOfParticles = (unsigned int)atoi(argc[2]);

    if (argv >= 4)
        numberOfHiddenNodes = (unsigned int)atoi(argc[3]);

    if (argv >= 5)
        numberOfMetropolisSteps = (unsigned int)pow(2, atof(argc[4]));

    if (argv >= 6)
        numberOfEquilibrationSteps = (unsigned int)pow(2, atof(argc[5]));

    if (argv >= 7)
        stepLength = (double)atof(argc[6]);

    if (argv >= 8)
        importanceSampling = (bool)atoi(argc[7]);

    if (argv >= 9)
        optimizerType = argc[8];

    if (argv >= 10)
        lr = (double)atof(argc[9]);

    if (argv >= 11)
        interaction = (bool)atoi(argc[10]);

    if (argv >= 12)
        filename = argc[11];
    if (argv >= 13)
        detailed = (bool)atoi(argc[12]);
    if (argv >= 14)
        beta1 = (double)atof(argc[13]);
    if (argv >= 15)
        beta2 = (double)atof(argc[14]);
    if (argv >= 16)
        eps = (double)atof(argc[15]);

    // start timing
    double start = omp_get_wtime();

    // Seed for the random number generator
    int seed = 42;

    auto state_rng = std::make_unique<Random>(rngType);
    auto solver_rng = std::make_unique<Random>(rngType);

    // Initialize particles
    auto particles = setupRandomUniformInitialState(
        omega, numberOfDimensions, numberOfParticles, *state_rng);

    // Construct a unique pointer to a new System
    std::unique_ptr<class Hamiltonian> hamiltonian;

    hamiltonian = std::make_unique<HarmonicOscillator>(omega, interaction);

    std::unique_ptr<class NeuralWaveFunction> wavefunction; // Empty wavefunction pointer constructor (can only be moved once)

    wavefunction = std::make_unique<GaussianBinary>(numberOfParticles, numberOfHiddenNodes, numberOfDimensions, std::move(state_rng));

    // Empty solver pointer, since it uses "rng" in its constructor
    std::unique_ptr<class MonteCarlo> solver;

    // Set what solver to use, pass on rng and additional parameters
    if (importanceSampling)
    {
        solver = std::make_unique<MetropolisHastings>(std::move(solver_rng), stepLength, D);
    }
    else
    {
        solver = std::make_unique<Metropolis>(std::move(solver_rng));
    }

    // Create system pointer, passing in all classes.
    auto system = std::make_unique<System>(
        // Construct unique_ptr to Hamiltonian
        std::move(hamiltonian),
        // Construct unique_ptr to wave function
        std::move(wavefunction),
        // Construct unique_ptr to solver, and move rng
        std::move(solver),
        // Move the vector of particles to system
        std::move(particles),
        // pass additional parameters
        importanceSampling,
        optimizerType,
        interaction);

    if (detailed)
    {
        system->saveSamples(filename + "_blocking.dat", 0);
    }

    // Run steps to equilibrate particles
    system->runEquilibrationSteps(stepLength, numberOfEquilibrationSteps);

    // Run the Metropolis algorithm
    std::unique_ptr<class Sampler> sampler;

    sampler = system->optimizeMetropolis(
        *system, filename, stepLength, numberOfMetropolisSteps, numberOfEquilibrationSteps, beta1, beta2, eps, lr, optimizerType);

    // Output information from the simulation, either as file or print
    sampler->output(*system, filename + ".txt", omega, optimizerType, importanceSampling, interaction);

    if (detailed)
    {
        system->saveFinalState(filename + "_Rs.txt");
    }
    // end timing
    double end = omp_get_wtime();
    std::cout << "Time elapsed " << rngType << ": " << end - start << std::endl;

    return 0;
}