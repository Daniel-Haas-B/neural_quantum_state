#pragma once

#include <memory>

#include "neuralwavefunction.h"

class GaussianBinary : public NeuralWaveFunction
{
public:
    GaussianBinary(int num_particles, int num_hidden_nodes, int num_dimensions, std::unique_ptr<class Random> rng); // a is the interaction parameter
    double evaluate(std::vector<std::unique_ptr<class Particle>> &particles);
    void computeParamDerivative(std::vector<std::unique_ptr<class Particle>> &particles,
                                std::vector<std::vector<std::vector<double>>> &weightDeltaPsi,
                                std::vector<std::vector<double>> &visDeltaPsi,
                                std::vector<double> &hidDeltaPsi);

    void quantumForce(std::vector<std::unique_ptr<class Particle>> &particles, Particle &particle, std::vector<double> &force);

    void setWeights(std::vector<std::vector<std::vector<double>>> weights);
    void setVisibleBias(std::vector<std::vector<double>> vis_bias);
    void setHiddenBias(std::vector<double> hidden_bias);

    std::vector<double> Qfac(std::vector<std::unique_ptr<class Particle>> &particles);
    double computeWeightNorms();

private:
    int m_numberOfParticles;
};
