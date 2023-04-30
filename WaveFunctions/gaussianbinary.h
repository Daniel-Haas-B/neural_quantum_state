#pragma once

#include <memory>

#include "neuralwavefunction.h"

class GaussianBinary : public NeuralWaveFunction
{
public:
    GaussianBinary(int num_particles, int num_hidden_nodes, std::unique_ptr<class Random> rng); // a is the interaction parameter
    double evaluate(std::vector<std::unique_ptr<class Particle>> &particles);
    // double evaluate_w(int proposed_particle_idx, class Particle &proposed_particle, class Particle &old_particle, std::vector<std::unique_ptr<class Particle>> &particles);

    std::vector<std::vector<double>> computeVisBiasDerivative(std::vector<std::unique_ptr<class Particle>> &particles);
    std::vector<double> computeHidBiasDerivative(std::vector<std::unique_ptr<class Particle>> &particles);
    std::vector<std::vector<std::vector<double>>> computeWeightDerivative(std::vector<std::unique_ptr<class Particle>> &particles);

    // double computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>> &particles);
    void quantumForce(std::vector<std::unique_ptr<class Particle>> &particles, Particle &particle, std::vector<double> &force);

    void setWeights(std::vector<std::vector<std::vector<double>>> weights);
    void setVisibleBias(std::vector<std::vector<double>> vis_bias);
    void setHiddenBias(std::vector<double> hidden_bias);

    std::vector<double> Qfac(std::vector<std::unique_ptr<class Particle>> &particles);

    double computeWeightNorms();

private:
    int m_numberOfParticles;
    // double u_p(double r_ij);  // u'(r_ij)
    // double u_pp(double r_ij); // u''(r_ij)
    // void grad_phi_ratio(std::vector<double> &v, Particle &particle, double alpha, double beta);
};
