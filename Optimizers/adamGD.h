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
        int numberOfHiddenNodes);

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
    int m_numberOfParticles;
    int m_numberOfDimensions;
    double m_beta1;
    double m_beta2;
    double m_epsilon;
};
