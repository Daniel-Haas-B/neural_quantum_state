#include "system.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <iomanip>
#include <string>

#include "Hamiltonians/hamiltonian.h"
#include "InitialStates/initialstate.h"
#include "Solvers/montecarlo.h"
#include "WaveFunctions/neuralwavefunction.h"
#include "Optimizers/optimizer.h"
#include "Optimizers/vanillaGD.h"
#include "Optimizers/momentumGD.h"
#include "Optimizers/adamGD.h"
#include "Optimizers/rmspropGD.h"
#include "particle.h"
#include "sampler.h"
#include <fstream>

using namespace std;
System::System(std::unique_ptr<class Hamiltonian> hamiltonian,
               std::unique_ptr<class NeuralWaveFunction> waveFunction,
               std::unique_ptr<class MonteCarlo> solver,
               std::vector<std::unique_ptr<class Particle>> particles,
               bool importSamples,
               string optimizerType,
               bool interaction)
{
  m_numberOfParticles = particles.size();
  m_numberOfDimensions = particles[0]->getNumberOfDimensions();
  m_numberOfHiddenNodes = waveFunction->getNumberOfHiddenNodes();
  m_hamiltonian = std::move(hamiltonian);
  m_waveFunction = std::move(waveFunction);
  m_solver = std::move(solver);
  m_particles = std::move(particles);

  m_importSamples = importSamples;
  m_optimizerType = optimizerType;
  m_interaction = interaction;
}

unsigned int System::runEquilibrationSteps(
    double stepLength, unsigned int numberOfEquilibrationSteps)
{
  unsigned int acceptedSteps = 0;

  for (unsigned int i = 0; i < numberOfEquilibrationSteps; i++)
  {
    acceptedSteps += m_solver->step(stepLength, *m_waveFunction, m_particles);
  }

  return acceptedSteps;
}

void System::runMetropolisSteps(
    std::unique_ptr<class Sampler> &sampler,
    double stepLength,
    unsigned int numberOfMetropolisSteps)
{
  // print visible bias inside the runMetropolisSteps
  std::vector<std::vector<double>> visibleBias = m_waveFunction->getVisibleBias();

  if (m_saveSamples)
    sampler->openSaveSample(m_saveSamplesFilename);

  bool acceptedStep;

  for (unsigned int i = 0; i < numberOfMetropolisSteps; i++)
  {
    /* Call solver method to do a single Monte-Carlo step.*/
    acceptedStep = m_solver->step(stepLength, *m_waveFunction, m_particles);

    // compute local energy
    sampler->sample(acceptedStep, this);
    if (m_saveSamples)
      sampler->saveSample(i);
  }

  sampler->computeAverages();
  if (m_saveSamples)
    sampler->closeSaveSample();
}

std::unique_ptr<class Sampler> System::optimizeMetropolis(
    System &system,
    std::string filename,
    double stepLength,
    unsigned int numberOfMetropolisSteps,
    unsigned int numberOfEquilibrationSteps,
    double beta1,
    double beta2,
    double epsilon,
    double learningRate,
    std::string optimizerType)
{

  int maxiter = 600;
  // run equilibration steps and store positions into vector
  runEquilibrationSteps(stepLength, numberOfEquilibrationSteps);

  for (unsigned int i = 0; i < m_numberOfParticles; i++)
  {
    m_particles[i]->saveEquilibrationPosition(); // by doind this, we just need to do equilibriation once in the GD
  }

  double decayRate = 0.99;
  double gamma = 0.05;

  std::unique_ptr<class Optimizer> optimizer;

  if (optimizerType == "vanillaGD")
  {
    optimizer = std::make_unique<VanillaGD>(
        learningRate,
        maxiter,
        stepLength,
        numberOfMetropolisSteps,
        m_numberOfHiddenNodes,
        m_numberOfDimensions,
        m_numberOfParticles);
  }
  else if (optimizerType == "momentumGD")
  {
    optimizer = std::make_unique<MomentumGD>(
        learningRate,
        gamma,
        maxiter,
        stepLength,
        numberOfMetropolisSteps,
        m_numberOfHiddenNodes,
        m_numberOfDimensions,
        m_numberOfParticles);
  }
  else if (optimizerType == "rmspropGD")
  {
    optimizer = std::make_unique<RmspropGD>(
        learningRate,
        decayRate,
        epsilon,
        maxiter,
        stepLength,
        numberOfMetropolisSteps,
        m_numberOfHiddenNodes,
        m_numberOfDimensions,
        m_numberOfParticles);
  }
  else if (optimizerType == "adamGD")
  {
    optimizer = std::make_unique<AdamGD>(
        learningRate,
        beta1,
        beta2,
        epsilon,
        maxiter,
        stepLength,
        numberOfMetropolisSteps,
        m_numberOfHiddenNodes,
        m_numberOfDimensions,
        m_numberOfParticles);
  }
  else
  {
    std::cout << "Optimizer type not recognized" << std::endl;
    exit(1);
  }

  // run optimizer
  auto sampler = optimizer->optimize(system, *m_waveFunction, m_particles, filename);
  std::cout << "TRUE OPTIMIZER END" << std::endl;

  return sampler;
}

void System::computeParamDerivative(std::vector<std::vector<std::vector<double>>> &weightDeltaPsi,
                                    std::vector<std::vector<double>> &visDeltaPsi,
                                    std::vector<double> &hidDeltaPsi)
{
  m_waveFunction->computeParamDerivative(m_particles, weightDeltaPsi, visDeltaPsi, hidDeltaPsi);
}

double System::computeLocalEnergy()
{
  // Helper function
  return m_hamiltonian->computeLocalEnergy(*m_waveFunction, m_particles);
}

void System::setWaveFunction(std::unique_ptr<class NeuralWaveFunction> waveFunction)
{
  m_waveFunction = std::move(waveFunction);
}

void System::setSolver(std::unique_ptr<class MonteCarlo> solver)
{
  m_solver = std::move(solver);
}

void System::saveSamples(std::string filename, int skip)
{
  // Tells system to save local energy estimates during run
  m_saveSamples = true;
  m_saveSamplesFilename = filename;

  // Due to powers of two being just a single bit 1, use bitwise AND to check if skip is a power of 2.
  bool isPow2 = false;
  if (skip == 0)
    isPow2 = true;
  else
    isPow2 = skip > 0 && !(skip & (skip - 1));

  assert(isPow2);
  m_skip = skip;
}

int System::getSkip()
{
  /*
    Returns the skip value used for saving samples.
  */
  return m_skip;
}

void System::saveFinalState(std::string filename)
{
  /*
    Saves the final state (position) of the particles to file.
  */
  std::ofstream file(filename, std::ios::out | std::ios::trunc);

  int w = 20;
  file << setw(w) << "x" << setw(w) << "y" << setw(w) << "z\n";

  for (unsigned int i = 0; i < m_numberOfParticles; i++)
  {
    auto r = m_particles.at(i)->getPosition();
    file << setw(w) << r.at(0) << setw(w) << r.at(1) << setw(w) << r.at(2) << "\n";
  }
  file.close();
}