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
#include "particle.h"
#include "sampler.h"
#include <fstream>

using namespace std;
System::System(std::unique_ptr<class Hamiltonian> hamiltonian,
               std::unique_ptr<class NeuralWaveFunction> waveFunction,
               std::unique_ptr<class MonteCarlo> solver,
               std::vector<std::unique_ptr<class Particle>> particles,
               bool importSamples,
               bool analytical,
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
  m_analytical = analytical;
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

std::unique_ptr<class Sampler> System::runMetropolisSteps(
    double stepLength, unsigned int numberOfMetropolisSteps)
{
  auto sampler = std::make_unique<Sampler>(m_numberOfParticles,
                                           m_numberOfDimensions,
                                           m_numberOfHiddenNodes,
                                           stepLength, numberOfMetropolisSteps);

  std::cout << "DEBUG runMetropolisSteps, m_numberOfParticles= " << m_numberOfParticles << std::endl;
  if (m_saveSamples)
    sampler->openSaveSample(m_saveSamplesFilename);

  for (unsigned int i = 0; i < numberOfMetropolisSteps; i++)
  {
    /* Call solver method to do a single Monte-Carlo step.*/
    bool acceptedStep = m_solver->step(stepLength, *m_waveFunction, m_particles);

    // compute local energy
    sampler->sample(acceptedStep, this);

    if (m_saveSamples)
      sampler->saveSample(i);
  }

  double lambda_l2 = 0.01;
  double cumWeight2 = m_waveFunction->computeWeightNorms();

  sampler->computeAverages(cumWeight2, lambda_l2);
  if (m_saveSamples)
    sampler->closeSaveSample();

  return sampler;
}

std::unique_ptr<class Sampler> System::optimizeMetropolis(
    System &system,
    std::string filename,
    double stepLength,
    unsigned int numberOfMetropolisSteps,
    unsigned int numberOfEquilibrationSteps,
    double epsilon,
    double learningRate)
{

  int maxiter = 20;
  // run equilibration steps and store positions into vector
  runEquilibrationSteps(stepLength, numberOfEquilibrationSteps);

  for (unsigned int i = 0; i < m_numberOfParticles; i++)
  {
    m_particles[i]->saveEquilibrationPosition(); // by doind this, we just need to do equilibriation once in the GD
  }
  // instantiate base class optimizer and cast to derived class VanillaGD

  // auto optimizer = std::make_unique<VanillaGD>(
  //     learningRate,
  //     maxiter,
  //     stepLength,
  //     numberOfMetropolisSteps,
  //     m_numberOfHiddenNodes);

  double beta1 = 0.9;
  double beta2 = 0.999;
  double epsilon2 = 1e-8;

  // auto optimizer = std::make_unique<VanillaGD>(
  //     learningRate,
  //     maxiter,
  //     stepLength,
  //     numberOfMetropolisSteps,
  //     m_numberOfHiddenNodes);

  auto optimizer = std::make_unique<AdamGD>(
      learningRate,
      beta1,
      beta2,
      epsilon2,
      maxiter,
      stepLength,
      numberOfMetropolisSteps,
      m_numberOfHiddenNodes);

  // run optimizer
  auto sampler = optimizer->optimize(system, *m_waveFunction, m_particles, filename);
  return sampler;
}

std::vector<std::vector<double>> System::computeVisBiasDerivative()
{
  // helper function to compute the derivative of the visible bias
  return m_waveFunction->computeVisBiasDerivative(m_particles);
}

std::vector<double> System::computeHidBiasDerivative()
{
  return m_waveFunction->computeHidBiasDerivative(m_particles);
}

std::vector<std::vector<std::vector<double>>> System::computeWeightDerivative()
{
  return m_waveFunction->computeWeightDerivative(m_particles);
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