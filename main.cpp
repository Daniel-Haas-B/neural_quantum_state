#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "Hamiltonians/harmonicoscillator.h"
// #include "Hamiltonians/anharmonicoscillator.h"
#include "InitialStates/initialstate.h"
#include "Math/random.h"
#include "Solvers/metropolis.h"
#include "Solvers/metropolishastings.h"
//   #include "WaveFunctions/simplegaussian.h"
#include "WaveFunctions/gaussianbinary.h"

// #include "WaveFunctions/interactinggaussian.h"
#include "WaveFunctions/neuralwavefunction.h"
#include "particle.h"
#include "sampler.h"
#include "system.h"
#include <omp.h>

using namespace std;

int main(int argv, char **argc)
{
    // Set default paramters
    int numberOfWalkers = 1;

    // Set default paramters
    unsigned int numberOfDimensions = 2;
    unsigned int numberOfParticles = 2;
    unsigned int numberOfMetropolisSteps = (unsigned int)pow(2, 20);
    unsigned int numberOfEquilibrationSteps = (unsigned int)pow(2, 20);
    double omega = 1.0;                                 // Oscillator frequency.
    double stepLength = 0.6;                            // Metropolis step length.
    double epsilon = 0.05;                              // Tolerance for gradient descent.
    double lr = 0.02;                                   // Learning rate for gradient descent.
    double interactionTerm = 0.0043 * sqrt(1. / omega); // Interection constant a. Notice that hbar = m = 1.

    // double dx = 1e-4;
    bool importanceSampling = false;
    bool analytical = true;
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
        cout << "analytical?, bool: If the analytical expression should be used. Defaults to true" << endl;
        cout << "interaction?, bool: If the interacting gaussian should be used. Defaults to false" << endl;
        cout << "filename, string: If the results should be dumped to a file, give the file name. If none is given, a simple print is performed." << endl;
        cout << "detailed?, bool: If the results should be printed in detail. Defaults to false" << endl;
        return 0;
    }

    // Check how many arguments are given and overwrite defaults. Works serially,
    // meaning if 4 parameters are given the first 4 paramters will be
    // overwritten, the rest will be defaults.
    if (argv >= 2)
        numberOfDimensions = (unsigned int)atoi(argc[1]);
    if (argv >= 3)
        numberOfParticles = (unsigned int)atoi(argc[2]);
    if (argv >= 4)
        numberOfMetropolisSteps = (unsigned int)pow(2, atof(argc[3]));
    if (argv >= 5)
        numberOfEquilibrationSteps = (unsigned int)pow(2, atof(argc[4]));
    if (argv >= 6)
        stepLength = (double)atof(argc[5]);
    if (argv >= 7)
        importanceSampling = (bool)atoi(argc[6]);
    if (argv >= 8)
        analytical = (bool)atoi(argc[7]);
    if (argv >= 9)
        lr = (double)atof(argc[8]);
    if (argv >= 10)
        interaction = (bool)atoi(argc[9]);
    if (argv >= 11)
        filename = argc[10];
    if (argv >= 12)
        detailed = (bool)atoi(argc[11]);

#pragma omp parallel for firstprivate(lr, filename, filename_samples, filename_posistions, numberOfWalkers)
    for (int i = 0; i < numberOfWalkers; i++)
    {
        int thread_id = omp_get_thread_num();
        std::cout << "STARTING WALK FROM THREAD " << thread_id << std::endl;

        if (numberOfWalkers > 1)
        {
            filename = filename + "thread_" + to_string(thread_id);
        }

        // Seed for the random number generator
        int seed = 2024 * (thread_id + 1);

        // The random engine can also be built without a seed
        auto state_rng = std::make_unique<Random>(seed);

        // another rng
        auto solver_rng = std::make_unique<Random>(seed);

        // Initialize particles
        auto particles = setupRandomUniformInitialState(
            omega, numberOfDimensions, numberOfParticles, *state_rng, interactionTerm);

        // Construct a unique pointer to a new System
        std::unique_ptr<class Hamiltonian> hamiltonian;

        // if (beta != 1.0)
        //{
        //     // for now throw error if beta != 1.0
        //
        //    throw std::invalid_argument("Interaction is not implemented for beta != 1.0");
        //
        //    // double gamma = beta;
        //    // hamiltonian = std::make_unique<AnharmonicOscillator>(gamma);
        //}
        // else
        //{
        hamiltonian = std::make_unique<HarmonicOscillator>(omega, interaction);
        //}

        std::unique_ptr<class NeuralWaveFunction> wavefunction; // Empty wavefunction pointer constructor (can only be moved once)

        int numberOfhiddenNodes = 2; // here for now
        wavefunction = std::make_unique<GaussianBinary>(numberOfParticles, numberOfhiddenNodes, std::move(state_rng));

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
            analytical,
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
            *system, filename, stepLength, numberOfMetropolisSteps, numberOfEquilibrationSteps, epsilon, lr);
        // Output information from the simulation, either as file or print
        sampler->output(*system, filename + ".txt", omega, analytical, importanceSampling, interaction);

        if (detailed)
        {
            system->saveFinalState(filename + "_Rs.txt");
        }
    }
    return 0;
}