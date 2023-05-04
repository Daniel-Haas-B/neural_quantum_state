#pragma once

#include <memory>
#include "WaveFunctions/neuralwavefunction.h"
#include "WaveFunctions/wavefunction.h"
#include "system.h"
#include "optimizer.h"

class AdamGD : public Optimizer
{
public:
    AdamGD(
        double learningRate,
        double beta1,
        double beta2,
        double epsilon,
        int maxIter,
        double stepLength,
        int numberOfMetropolisSteps,
        int numberOfHiddenNodes,
        int numberOfDimensions,
        int numberOfParticles);

    // optimize will return sampler

    std::unique_ptr<class Sampler> optimize(
        System &system,
        class NeuralWaveFunction &waveFunction,
        std::vector<std::unique_ptr<class Particle>> &particles,
        std::string filename);

public:
    double m_beta1;
    double m_beta2;
    double m_epsilon;
};
