#include <memory>
#include <vector>

#include "metropolishastings.h"
#include "WaveFunctions/neuralwavefunction.h"
#include "particle.h"
#include "Math/random.h"

#include <iostream>

MetropolisHastings::MetropolisHastings(std::unique_ptr<class Random> rng, double timeStep, double D)
    : MonteCarlo(std::move(rng))
{
    m_timeStep = timeStep;
    m_sqrtTimeStep = std::sqrt(timeStep);
    m_D = D; // Diffusion constant
}

bool MetropolisHastings::step(
    double stepLength,
    class NeuralWaveFunction &waveFunction,
    std::vector<std::unique_ptr<class Particle>> &particles)
{
    /*
    Here the step related to Metroplis-Hasting algo should be performed, using importance sampling.
    */
    int numberOfParticles = particles.size();
    int numberOfDimensions = particles.at(0)->getNumberOfDimensions();

    std::vector<double> qForceOld(numberOfDimensions);
    std::vector<double> qForceNew(numberOfDimensions);

    int proposed_particle_idx = m_rng->nextInt(0, numberOfParticles - 1); // Choose a particle at random

    Particle &proposed_particle = *particles.at(proposed_particle_idx); // Get a reference to the particle
    Particle old_particle = proposed_particle;

    double Psi_old = waveFunction.evaluate(particles);
    waveFunction.quantumForce(particles, proposed_particle, qForceOld); // if wave function is not interecting, particles will be ignored

    for (int q = 0; q < numberOfDimensions; q++)
    {
        proposed_particle.adjustPosition(
            m_D * qForceOld.at(q) * m_timeStep + m_rng->nextGaussian(0.0, 1.0) * m_sqrtTimeStep, q); // Importance sampling. This formula is similar to the Metropolis algorithm (.adjustPosition(stepLength * (m_rng->nextDouble() - .5), q)), but with the addition of the quantum force term.
    }

    double Psi_new = waveFunction.evaluate(particles);
    waveFunction.quantumForce(particles, proposed_particle, qForceNew); // Calculate the quantum force at the new position

    double G_ratio = 0; // The expoents of the ratio of the Greens functions
    double old_p_q, new_p_q;
    for (int q = 0; q < numberOfDimensions; q++)
    {
        old_p_q = old_particle.getPosition().at(q);
        new_p_q = proposed_particle.getPosition().at(q);
        G_ratio += 0.5 * (qForceOld.at(q) + qForceNew.at(q)) * (old_p_q - new_p_q + 0.5 * m_D * m_timeStep * (qForceOld.at(q) - qForceNew.at(q)));
    }

    G_ratio = std::exp(G_ratio);
    double w = (Psi_new * Psi_new) / (Psi_old * Psi_old); // Metropolis test
    w *= G_ratio;                                         // multiply with the ratio of the Greens functions

    // FAST OPTION
    // double w = G_ratio * waveFunction.evaluate_w(proposed_particle_idx, // this is a fast way of doing things
    //                                              proposed_particle,
    //                                              old_particle,
    //                                              particles); // Metropolis test
    if (w >= m_rng->nextDouble()) // Accept the step
    {
        return true;
    }
    else
    {
        *particles.at(proposed_particle_idx) = old_particle; // Reject the step and restore the old position
        return false;
    }
}