#include <memory>
#include <cmath>
#include <cassert>
#include "Math/random.h"

#include "gaussianbinary.h"
#include "neuralwavefunction.h"
#include "../system.h"
#include "../particle.h"
#include <Eigen/Dense>

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

    double bias_random_scale = 0.0001;
    double weight_random_scale = 1 / sqrt(m_numberOfHiddenNodes);

    // Initialize the hidden bias
    m_hiddenBias = std::vector<double>(m_numberOfHiddenNodes, 0.0);
    for (int i = 0; i < m_numberOfHiddenNodes; i++)
    {
        m_hiddenBias[i] = m_rng->nextGaussian(0, bias_random_scale);
    }

    // Initialize the weights
    m_weights = std::vector<std::vector<std::vector<double>>>(m_numberOfVisibleNodes, std::vector<std::vector<double>>(m_numberOfDimensions, std::vector<double>(m_numberOfHiddenNodes, 0.0)));
    m_visibleBias = std::vector<std::vector<double>>(m_numberOfVisibleNodes, std::vector<double>(m_numberOfDimensions, 0.0));

    for (int i = 0; i < m_numberOfVisibleNodes; i++)
    {
        for (int j = 0; j < m_numberOfDimensions; j++)
        {
            m_visibleBias[i][j] = m_rng->nextGaussian(0, bias_random_scale);
            for (int k = 0; k < m_numberOfHiddenNodes; k++)
            {
                m_weights[i][j][k] = m_rng->nextGaussian(0, weight_random_scale);
            }
        }
    }

    std ::cout << "GaussianBinary initialized" << std ::endl;
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

    // First term (gaussian part)
    for (int i = 0; i < m_numberOfVisibleNodes; i++) // this is the same as number of particles
    {
        for (int j = 0; j < m_numberOfDimensions; j++)
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
    for (unsigned int i = 0; i < m_numberOfVisibleNodes; i++) // this is the same as number of particles
    {
        for (unsigned int j = 0; j < numberOfDimensions; j++)
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
    double weights_ijk = 0.0;
    for (int i = 0; i < m_numberOfParticles; i++)
    {
        for (int j = 0; j < m_numberOfDimensions; j++)
        {
            for (int k = 0; k < m_numberOfHiddenNodes; k++)
            {
                weights_ijk = m_weights[i][j][k];
                weightNorm2 += std::sqrt(weights_ijk * weights_ijk);
            }
        }
    }

    return weightNorm2;
}

void GaussianBinary::computeParamDerivative(std::vector<std::unique_ptr<class Particle>> &particles,
                                            std::vector<std::vector<std::vector<double>>> &weightDeltaPsi,
                                            std::vector<std::vector<double>> &visDeltaPsi,
                                            std::vector<double> &hidDeltaPsi)
{

    double sigma2 = m_sigma * m_sigma;
    std::vector<double> Q = Qfac(particles);

    // visible
    for (int i = 0; i < m_numberOfParticles; i++)
    {
        for (int j = 0; j < m_numberOfDimensions; j++)
        {
            visDeltaPsi[i][j] = (particles[i]->getPosition()[j] - m_visibleBias[i][j]) / sigma2;
        }
    }

    // hidden
    for (int i = 0; i < m_numberOfHiddenNodes; i++)
    {
        hidDeltaPsi[i] = 1 / (1 + exp(-Q[i]));
    }

    // weights

    for (int i = 0; i < m_numberOfParticles; i++)
    {
        for (int j = 0; j < m_numberOfDimensions; j++)
        {
            for (int k = 0; k < m_numberOfHiddenNodes; k++)
            {
                weightDeltaPsi[i][j][k] = m_weights[i][j][k] / (sigma2 * (1 + std::exp(-Q[k])));
            }
        }
    }
}

// std::vector<std::vector<double>> GaussianBinary::computeVisBiasDerivative(std::vector<std::unique_ptr<class Particle>> &particles)
// {
//     // a,b,w we already have from private, but the position needs to be passed via particles
//     // particles will carry the old position

//     std::vector<std::vector<double>> wf_der = std::vector<std::vector<double>>(m_numberOfParticles, std::vector<double>(m_numberOfDimensions, 0.0));
//     double sigma2 = m_sigma * m_sigma;

//     for (int i = 0; i < m_numberOfParticles; i++)
//     {
//         for (int j = 0; j < m_numberOfDimensions; j++)
//         {
//             wf_der[i][j] = (particles[i]->getPosition()[j] - m_visibleBias[i][j]) / sigma2;
//         }
//     }

//     return wf_der;
// }

// std::vector<double> GaussianBinary::computeHidBiasDerivative(std::vector<std::unique_ptr<class Particle>> &old_particles)
// {
//     std::vector<double> wf_der = std::vector<double>(m_numberOfHiddenNodes, 0.0);
//     std::vector<double> Q = Qfac(old_particles);

//     for (int i = 0; i < m_numberOfHiddenNodes; i++)
//     {
//         wf_der[i] = 1 / (1 + exp(-Q[i]));
//     }

//     return wf_der;
// }

// std::vector<std::vector<std::vector<double>>> GaussianBinary::computeWeightDerivative(std::vector<std::unique_ptr<class Particle>> &old_particles)
// {
//     std::vector<std::vector<std::vector<double>>> wf_der = std::vector<std::vector<std::vector<double>>>(m_numberOfParticles, std::vector<std::vector<double>>(m_numberOfDimensions, std::vector<double>(m_numberOfHiddenNodes, 0.0)));
//     std::vector<double> Q = Qfac(old_particles);

//     double sigma2 = m_sigma * m_sigma;

//     for (int i = 0; i < m_numberOfParticles; i++)
//     {
//         for (int j = 0; j < m_numberOfDimensions; j++)
//         {
//             for (int k = 0; k < m_numberOfHiddenNodes; k++)
//             {
//                 wf_der[i][j][k] = m_weights[i][j][k] / (sigma2 * (1 + std::exp(-Q[k])));
//             }
//         }
//     }

//     return wf_der;
// }

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