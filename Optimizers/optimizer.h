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
        int numberOfHiddenNodes);
    virtual ~Optimizer() = default;

    virtual std::unique_ptr<class Sampler> optimize(
        System &system,
        class NeuralWaveFunction &waveFunction,
        std::vector<std::unique_ptr<class Particle>> &particles,
        std::string filename) = 0;

    virtual std::vector<double> computeGradientNorms(
        std::vector<double> hidEnergyDer,
        std::vector<std::vector<double>> visEnergyDer,
        std::vector<std::vector<std::vector<double>>> weightEnergyDer) = 0;

protected:
    double m_learningRate;
    int m_maxIter;
    double m_stepLength;
    int m_numberOfMetropolisSteps;
    int m_numberOfHiddenNodes;
};
