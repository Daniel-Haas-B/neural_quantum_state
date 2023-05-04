#pragma once

#include <vector>
#include <memory>

#include "system.h"

class Optimizer
{
    // this class will hold methods like gradient descent and stochastic gradient descent with momentum
public:
    // must have max iter, learning rate and type of optimizer
    Optimizer(
        double learningRate,
        int maxIter,
        double stepLength,
        int numberOfMetropolisSteps,
        int numberOfHiddenNodes,
        int numberOfDimensions,
        int numberOfParticles);
    virtual ~Optimizer() = default;

    virtual std::unique_ptr<class Sampler> optimize(
        System &system,
        class NeuralWaveFunction &waveFunction,
        std::vector<std::unique_ptr<class Particle>> &particles,
        std::string filename) = 0;

    std::vector<double> computeGradientNorms(
        std::vector<double> hidEnergyDer,
        std::vector<std::vector<double>> visEnergyDer,
        std::vector<std::vector<std::vector<double>>> weightEnergyDer);

protected:
    double m_learningRate;
    unsigned int m_maxIter;
    double m_stepLength;
    unsigned int m_numberOfMetropolisSteps;
    unsigned int m_numberOfHiddenNodes;
    unsigned int m_numberOfParticles;
    unsigned int m_numberOfDimensions;
};
