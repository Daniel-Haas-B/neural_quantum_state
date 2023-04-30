#include <memory>
#include <cmath>
#include <cassert>
#include "Math/random.h"

#include "gaussianbinary.h"
#include "neuralwavefunction.h"
#include "../system.h"
#include "../particle.h"

#include <iostream>
using namespace std;

GaussianBinary::GaussianBinary(int num_particles, int num_hidden_nodes, std::unique_ptr<class Random> rng)
    : NeuralWaveFunction(std::move(rng))
{
    m_numberOfVisibleNodes = num_particles;
    m_numberOfParticles = num_particles;
    m_numberOfHiddenNodes = num_hidden_nodes;
    m_numberOfDimensions = 2;

    m_sigma = 1.0;
    // m_visibleBias is a matrix of size numberOfVisibleNodes x m_numberOfDimensions

    double bias_random_scale = 0.1;
    double weight_random_scale = 1 / sqrt(m_numberOfHiddenNodes);

    // Initialize the visible bias
    m_visibleBias.resize(m_numberOfVisibleNodes); // resize the vector to have m_numberOfVisibleNodes elements
    m_hiddenBias.resize(m_numberOfHiddenNodes);   // resize the vector to have m_numberOfHiddenNodes elements
    m_weights.resize(m_numberOfVisibleNodes);     // resize the vector to have m_numberOfVisibleNodes elements
    for (int i = 0; i < m_numberOfVisibleNodes; i++)
    {
        m_weights[i].resize(m_numberOfDimensions); // resize the vector to have m_numberOfDimensions elements
        for (int j = 0; j < m_numberOfDimensions; j++)
        {
            m_visibleBias[i][j] = m_rng->nextGaussian(0, bias_random_scale);

            m_weights[i][j].resize(m_numberOfHiddenNodes); // resize the vector to have m_numberOfHiddenNodes elements
            for (int k = 0; k < m_numberOfHiddenNodes; k++)
            {
                m_weights[i][j][k] = m_rng->nextGaussian(0, weight_random_scale);
                m_hiddenBias[k] = m_rng->nextGaussian(0, bias_random_scale);
            }
        }
    }
}

double GaussianBinary::evaluate(std::vector<std::unique_ptr<class Particle>> &particles)
{
    /*
     * w_ij is the weight matrix that connects the visible and hidden layers.
     * b_j is the HIDDEN layer bias
     * a_i is the VISIBLE layer bias
     * h_j is the HIDDEN layer activation
     * sigma_i is the sqrt of the variance of the hidden layer
     */
    double sigma2 = m_sigma * m_sigma;
    double psi1 = 0.0;
    double psi2 = 1.0;
    double a = 0.0;
    double r = 0.0;

    // vector Q is the vector of the hidden layer activations
    std::vector<double> Q = Qfac(particles);

    static const int numberOfDimensions = particles.at(0)->getNumberOfDimensions();

    // First term (gaussian part)
    for (int i = 0; i < m_numberOfVisibleNodes; i++) // this is the same as number of particles
    {
        for (int j = 0; j < numberOfDimensions; j++)
        {
            r = particles[i]->getPosition()[j];
            a = m_visibleBias[i][j];
            psi1 += (r - a) * (r - a);
        }
    }
    psi1 = exp(-psi1 / (2 * sigma2));

    // Second term (interaction part)
    for (int i = 0; i < m_numberOfHiddenNodes; i++)
    {
        psi2 *= (1 + exp(Q[i]));
    }

    return psi1 * psi2;
}

std::vector<double> GaussianBinary::Qfac(std::vector<std::unique_ptr<class Particle>> &particles)
{
    double r = 0.0;
    double sigma2 = m_sigma * m_sigma;
    std::vector<double> Q = std::vector<double>(m_numberOfHiddenNodes, 0.0);
    std::vector<double> Qtemp = std::vector<double>(m_numberOfHiddenNodes, 0.0);

    static const int numberOfDimensions = particles.at(0)->getNumberOfDimensions();

    // Calculate the sum of the products
    for (int i = 0; i < m_numberOfVisibleNodes; i++) // this is the same as number of particles
    {
        for (int j = 0; j < numberOfDimensions; j++)
        {
            r = particles[i]->getPosition()[j];
            for (int k = 0; k < m_numberOfHiddenNodes; k++)
            {

                Qtemp[k] += m_weights[i][j][k] * r;
            }
        }
    }

    for (int j = 0; j < m_numberOfHiddenNodes; j++)
    {
        Q[j] = m_hiddenBias[j] + Qtemp[j] / sigma2;
    }

    return Q;
}

double GaussianBinary::computeWeightNorms()
{
    double weightNorm2 = 0.0;
    for (int i = 0; i < m_numberOfParticles; i++)
    {
        for (int j = 0; j < m_numberOfDimensions; j++)
        {
            for (int k = 0; k < m_numberOfHiddenNodes; k++)
            {
                weightNorm2 += m_weights[i][j][k] * m_weights[i][j][k];
            }
        }
    }

    return weightNorm2;
}

// double GaussianBinary::evaluate_w(int proposed_particle_idx, class Particle &proposed_particle, class Particle &old_particle, std::vector<std::unique_ptr<class Particle>> &particles)
//{
//     /*
//      This is the wave function ratio for the Metropolis algorithm.
//      It is a clever way to avoid having to evaluate the wave function for all particles at each step.
//      The gaussian part is still present, but we also have to recalculate every term where the proposed particle is present (one N product with f_ij)
//     */
//     // static const int numberOfDimensions = particles.at(0)->getNumberOfDimensions();
//     static const double a = m_interactionTerm;
//     double alpha = m_parameters.at(0);
//     double beta = m_parameters.at(1);
//
//     double r2_proposed, r2_old;
//     r2_proposed = 0;
//     r2_old = 0;
//
//     r2_proposed = particle_r2(proposed_particle);
//     r2_old = particle_r2(old_particle);
//
//     // beta corrections to r2. Notice this lets us use the same r2, even if beta is not 1
//     r2_proposed += (proposed_particle.getPosition()[2] * proposed_particle.getPosition()[2]) * (beta - 1);
//     r2_old += (old_particle.getPosition()[2] * old_particle.getPosition()[2]) * (beta - 1);
//
//     double gaussian = std::exp(-2.0 * alpha * (r2_proposed - r2_old)); // Same as non-interactive
//
//     double interaction = 1;
//     double r_gj_prime = 0; // |r_g - r_j| in proposed R configuration
//     double r_gj = 0;       // |r_g - r_j| in old R configuration
//     double delta = 0;      // If any particle distances are less than a, evalute interaction term to 0
//
//     // proposed_idx != i product. Divided into two loops to avoid if statments
//     for (int i = 0; i < proposed_particle_idx; i++)
//     {
//         r_gj_prime = std::sqrt(particle_r2(proposed_particle, *particles[i]));
//         r_gj = std::sqrt(particle_r2(old_particle, *particles[i]));
//         delta = (r_gj_prime > a) * (r_gj > a);
//         if (!delta)
//             return 0;
//         interaction *= (1.0 - a / r_gj_prime) / (1.0 - a / r_gj); // ratio for relative r_gj distance
//     }
//     // Same as above but for the indicies after proposed_particle_idx
//     for (unsigned int i = proposed_particle_idx + 1; i < m_numberOfParticles; i++)
//     {
//         r_gj_prime = std::sqrt(particle_r2(proposed_particle, *particles[i]));
//         r_gj = std::sqrt(particle_r2(old_particle, *particles[i]));
//         delta = (r_gj_prime > a) * (r_gj > a);
//         if (!delta)
//             return 0;
//         interaction *= (1.0 - a / r_gj_prime) / (1.0 - a / r_gj);
//     }
//
//     return gaussian * interaction * interaction; // Dont forget to square the interaction part :)
// }

std::vector<std::vector<double>> GaussianBinary::computeVisBiasDerivative(std::vector<std::unique_ptr<class Particle>> &particles)
{
    // a,b,w we already have from private, but the position needs to be passed via particles
    // particles will carry the old position

    std::vector<std::vector<double>> wf_der = std::vector<std::vector<double>>(m_numberOfParticles, std::vector<double>(m_numberOfDimensions, 0.0));
    double sigma2 = m_sigma * m_sigma;

    for (int i = 0; i < m_numberOfParticles; i++)
    {
        for (int j = 0; j < m_numberOfDimensions; j++)
        {
            wf_der[i][j] = (particles[i]->getPosition()[j] - m_visibleBias[i][j]) / sigma2;
        }
    }

    return wf_der;
}

std::vector<double> GaussianBinary::computeHidBiasDerivative(std::vector<std::unique_ptr<class Particle>> &old_particles)
{
    std::vector<double> wf_der = std::vector<double>(m_numberOfHiddenNodes, 0.0);
    std::vector<double> Q = Qfac(old_particles);

    for (int i = 0; i < m_numberOfHiddenNodes; i++)
    {
        wf_der[i] = 1 / (1 + exp(-Q[i]));
    }

    return wf_der;
}

std::vector<std::vector<std::vector<double>>> GaussianBinary::computeWeightDerivative(std::vector<std::unique_ptr<class Particle>> &old_particles)
{
    std::vector<std::vector<std::vector<double>>> wf_der = std::vector<std::vector<std::vector<double>>>(m_numberOfParticles, std::vector<std::vector<double>>(m_numberOfDimensions, std::vector<double>(m_numberOfHiddenNodes, 0.0)));
    std::vector<double> Q = Qfac(old_particles);

    double sigma2 = m_sigma * m_sigma;

    for (int i = 0; i < m_numberOfParticles; i++)
    {
        for (int j = 0; j < m_numberOfDimensions; j++)
        {
            for (int k = 0; k < m_numberOfHiddenNodes; k++)
            {
                wf_der[i][j][k] = m_weights[i][j][k] / (sigma2 * (1 + std::exp(-Q[k])));
            }
        }
    }

    return wf_der;
}

void GaussianBinary::quantumForce(std::vector<std::unique_ptr<class Particle>> &particles, Particle &particle, std::vector<double> &force)
{
    static const int numberOfDimensions = particle.getNumberOfDimensions();

    static const double sigma2 = m_sigma * m_sigma;

    std::vector<double> Q = Qfac(particles);

    double sum1 = 0;

    for (int i = 0; i < m_numberOfVisibleNodes; i++)
    {
        for (int j = 0; j < numberOfDimensions; j++)
        {
            for (int k = 0; k < m_numberOfHiddenNodes; k++)
            {
                sum1 += m_weights[i][j][k] / (1 + std::exp(-Q[k]));
            }
        }
    }

    for (int q = 0; q < numberOfDimensions; q++)
    {
        force.at(q) = 2 * (-particle.getPosition().at(q) / sigma2 + sum1 / sigma2);
    }
}

void GaussianBinary::setVisibleBias(std::vector<std::vector<double>> visibleBias)
{
    m_visibleBias = visibleBias;
}

void GaussianBinary::setHiddenBias(std::vector<double> hiddenBias)
{
    m_hiddenBias = hiddenBias;
}

void GaussianBinary::setWeights(std::vector<std::vector<std::vector<double>>> weights)
{
    m_weights = weights;
}