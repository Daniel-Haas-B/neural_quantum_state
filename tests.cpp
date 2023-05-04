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

using namespace std;

int main(int argv, char **argc)
{
    // Set default paramters
    unsigned int numberOfDimensions = 2;
    unsigned int numberOfParticles = 2;
    unsigned int numberOfHiddenNodes = 2;
    double omega = 1.0;      // Oscillator frequency.
    double stepLength = 0.6; // Metropolis step length.

    // double dx = 1e-4;
    bool importanceSampling = false;
    bool analytical = true;
    double D = 0.5;
    string filename = "";
    string filename_samples = "";
    string filename_posistions = "";
    bool interaction = false;

    if (argv >= 2)
        numberOfDimensions = (unsigned int)atoi(argc[1]);
    if (argv >= 3)
        numberOfParticles = (unsigned int)atoi(argc[2]);
    if (argv >= 4)
        numberOfHiddenNodes = (unsigned int)atoi(argc[3]);
    if (argv >= 5)
        interaction = (bool)atoi(argc[4]);

    // Seed for the random number generator
    int seed = 2024;

    // The random engine can also be built without a seed
    auto state_rng = std::make_unique<Random>(seed);

    // another rng
    auto solver_rng = std::make_unique<Random>(seed);

    // Initialize particles
    auto particles = setupRandomUniformInitialState(
        omega, numberOfDimensions, numberOfParticles, *state_rng);

    // Construct a unique pointer to a new System
    std::unique_ptr<class Hamiltonian> hamiltonian;

    hamiltonian = std::make_unique<HarmonicOscillator>(omega, interaction);

    std::unique_ptr<class NeuralWaveFunction> wavefunction; // Empty wavefunction pointer constructor (can only be moved once)

    wavefunction = std::make_unique<GaussianBinary>(numberOfParticles, numberOfHiddenNodes, std::move(state_rng));

    // set specific parameters for the neural network

    std::vector<std::vector<double>> a = wavefunction->getVisibleBias();
    std::vector<std::vector<std::vector<double>>> w = wavefunction->getWeights();
    std::vector<double> hiddenBias = wavefunction->getHiddenBias();

    std::cout << "a: " << a[0][0] << endl;
    std::cout << "w: " << w[0][0][0] << endl;
    std::cout << "b: " << hiddenBias[0] << endl;

    /// this is probably the ugliest thing I have ever done

    ///////// TEST QFAC
    std::ofstream afile;
    afile.open("../tests/Data/a.txt");
    afile.precision(16);

    for (unsigned int i = 0; i < a.size(); i++)
    {
        for (unsigned int j = 0; j < a[i].size(); j++)
        {
            afile << a[i][j] << " ";
        }
        afile << endl;
    }
    afile.close();

    std::ofstream wfile;
    wfile.open("../tests/Data/w.txt");
    wfile.precision(16);

    unsigned int n = w.size();
    unsigned int m = w[0].size();
    unsigned int p = w[0][0].size();

    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < m; j++)
        {
            for (unsigned int k = 0; k < p; k++)
            {
                wfile << w[i][j][k] << " ";
            }
            wfile << endl;
        }
        wfile << endl;
    }
    wfile.close();

    std::ofstream bfile;
    bfile.open("../tests/Data/b.txt");
    bfile.precision(16);

    for (unsigned int i = 0; i < hiddenBias.size(); i++)
    {
        bfile << hiddenBias[i] << endl;
    }
    bfile.close();

    std::vector<double> Q = wavefunction->Qfac(particles);

    // very dirty test of Qfac. This SHOULD BE DONE IN A FUNCTION IN THE FUTURE
    // output Q to a file
    std::ofstream Qfile;
    Qfile.open("../tests/Data/testQfac.txt");
    Qfile.precision(16);

    // also do it to the wf position

    std::ofstream wfr;
    wfr.open("../tests/Data/wfpos.txt");
    wfr.precision(16);

    // for particles and dimensions
    for (unsigned int i = 0; i < numberOfParticles; i++)
    {
        for (unsigned int j = 0; j < numberOfDimensions; j++)
        {
            wfr << particles[i]->getPosition()[j] << endl;
        }
        wfr << endl;
    }
    wfr.close();

    for (unsigned int i = 0; i < Q.size(); i++)
    {
        Qfile << Q[i] << endl;
    }
    Qfile.close();
    ///////// END TEST QFAC

    ///////// TEST WF EVAL

    // evaluate(std::vector<std::unique_ptr<class Particle>> &particles)
    double wf = wavefunction->evaluate(particles);

    std::ofstream wf_file;
    wf_file.open("../tests/Data/wf_eval.txt");
    wf_file.precision(16);

    wf_file << wf << endl;

    wf_file.close();

    ///////// END TEST WF EVAL

    ///////// TEST GRADIENT
    // std::vector<double> GaussianBinary::computeHidBiasDerivative(std::vector<std::unique_ptr<class Particle>> &old_particles)

    std::vector<double> hidBiasDer = wavefunction->computeHidBiasDerivative(particles);
    std::vector<std::vector<double>> visBiasDer = wavefunction->computeVisBiasDerivative(particles);
    std::vector<std::vector<std::vector<double>>> weightsDer = wavefunction->computeWeightDerivative(particles);

    std::ofstream hidBiasDer_file;
    hidBiasDer_file.open("../tests/Data/hidBiasDer.txt");
    hidBiasDer_file.precision(16);

    for (unsigned int i = 0; i < hidBiasDer.size(); i++)
    {
        hidBiasDer_file << hidBiasDer[i] << endl;
    }
    hidBiasDer_file.close();

    std::ofstream visBiasDer_file;
    visBiasDer_file.open("../tests/Data/visBiasDer.txt");
    visBiasDer_file.precision(16);

    for (unsigned int i = 0; i < visBiasDer.size(); i++)
    {
        for (unsigned int j = 0; j < visBiasDer[i].size(); j++)
        {
            visBiasDer_file << visBiasDer[i][j] << " ";
        }
        visBiasDer_file << endl;
    }
    visBiasDer_file.close();

    std::ofstream weightsDer_file;
    weightsDer_file.open("../tests/Data/weightsDer.txt");
    weightsDer_file.precision(16);

    for (unsigned int i = 0; i < weightsDer.size(); i++)
    {
        for (unsigned int j = 0; j < weightsDer[i].size(); j++)
        {
            for (unsigned int k = 0; k < weightsDer[i][j].size(); k++)
            {
                weightsDer_file << weightsDer[i][j][k] << " ";
            }
            weightsDer_file << endl;
        }
        weightsDer_file << endl;
    }
    weightsDer_file.close();

    ///// END TEST GRADIENT

    //// test local energu
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
    double localEnergy = system->computeLocalEnergy();

    std::ofstream localEnergy_file;
    localEnergy_file.open("../tests/Data/localEnergy.txt");
    localEnergy_file.precision(16);

    localEnergy_file << localEnergy << endl;

    localEnergy_file.close();

    // // Empty solver pointer, since it uses "rng" in its constructor
    // std::unique_ptr<class MonteCarlo> solver;

    // // Set what solver to use, pass on rng and additional parameters
    // if (importanceSampling)
    // {
    //     solver = std::make_unique<MetropolisHastings>(std::move(solver_rng), stepLength, D);
    // }
    // else
    // {
    //     solver = std::make_unique<Metropolis>(std::move(solver_rng));
    // }

    // // Create system pointer, passing in all classes.
    // auto system = std::make_unique<System>(
    //     // Construct unique_ptr to Hamiltonian
    //     std::move(hamiltonian),
    //     // Construct unique_ptr to wave function
    //     std::move(wavefunction),
    //     // Construct unique_ptr to solver, and move rng
    //     std::move(solver),
    //     // Move the vector of particles to system
    //     std::move(particles),
    //     // pass additional parameters
    //     importanceSampling,
    //     analytical,
    //     interaction);

    // // Run steps to equilibrate particles

    // system->runEquilibrationSteps(stepLength, numberOfEquilibrationSteps);

    // // Run the Metropolis algorithm
    // std::unique_ptr<class Sampler> sampler;

    // sampler = system->optimizeMetropolis(
    //     *system, filename, stepLength, numberOfMetropolisSteps, numberOfEquilibrationSteps, epsilon, lr);
    // // Output information from the simulation, either as file or print

    return 0;
}