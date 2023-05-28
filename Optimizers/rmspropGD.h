#pragma once

#include <memory>
#include "WaveFunctions/neuralwavefunction.h"
#include "WaveFunctions/wavefunction.h"
#include "system.h"
#include "optimizer.h"

class RmspropGD : public Optimizer
{
public:
    RmspropGD(
        double learningRate,
        double decayRate,
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

    std::vector<double> computeGradientNorms(
        std::vector<double> hidEnergyDer,
        std::vector<std::vector<double>> visEnergyDer,
        std::vector<std::vector<std::vector<double>>> weightEnergyDer);

public:
    double m_decayRate;
    double m_epsilon;
    std::vector<std ::vector<double>> m_msVisBias;
    std::vector<double> m_msHidBias;
    std::vector<std::vector<std::vector<double>>> m_msWeights;
};
