#include <memory>
#include <cassert>
#include <iostream>
#include <cmath>

#include "harmonicoscillator.h"
#include "../particle.h"
#include "../WaveFunctions/neuralwavefunction.h"

using std::cout;
using std::endl;

HarmonicOscillator::HarmonicOscillator(double omega, bool interaction)
{
    assert(omega > 0);
    m_omega = omega;
    m_interaction = interaction;
}

double HarmonicOscillator::computeLocalEnergy(
    class NeuralWaveFunction &waveFunction,
    std::vector<std::unique_ptr<class Particle>> &particles)
{
    double sigma = 1.0; // waveFunction.get_sigma();
    double sigma2 = sigma * sigma;

    double localEnergy = 0;
    int numParticles = particles.size();
    int numberOfDimensions = particles.at(0)->getNumberOfDimensions();
    int numberOfHiddenNodes = waveFunction.getNumberOfHiddenNodes();

    double sum1, sum2, dlnpsi1, dlnpsi2, expQ;
    double w_kqj;

    std::vector<std::vector<double>> a = waveFunction.getVisibleBias();
    std::vector<std::vector<std::vector<double>>> w = waveFunction.getWeights();
    std::vector<double> Q = waveFunction.Qfac(particles);

    for (int k = 0; k < numParticles; k++)
    {
        Particle &particle = *particles.at(k);

        for (int q = 0; q < numberOfDimensions; q++)
        {
            sum1 = 0;
            sum2 = 0;
            double r = particle.getPosition()[q];
            for (int j = 0; j < numberOfHiddenNodes; j++)
            {
                w_kqj = w[k][q][j];
                expQ = std::exp(Q[j]);
                sum1 += w_kqj / (1 + std::exp(-Q[j]));
                sum2 += w_kqj * w_kqj * expQ / ((1 + expQ) * (1 + expQ));
            }

            dlnpsi1 = (a[k][q] - r + sum1) / sigma2;
            dlnpsi2 = (-1 + sum2 / sigma2) / sigma2;
            localEnergy += 0.5 * (-dlnpsi1 * dlnpsi1 - dlnpsi2 + r * r);
        }
    }

    if (m_interaction)
    {
        assert(numParticles > 1);
        double r2_sum = 0;
        for (int k = 0; k < numParticles; k++)
        {
            Particle &particle1 = *particles.at(k);
            for (int l = 0; l < k; l++)
            {
                Particle &particle2 = *particles.at(l);
                r2_sum = particle_r2(particle1, particle2);
                localEnergy += 1.0 / sqrt(r2_sum);
            }
        }
    }

    return localEnergy;
}
