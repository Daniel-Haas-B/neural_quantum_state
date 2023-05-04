#include <memory>
#include <vector>

#include "rmspropGD.h"
#include "WaveFunctions/neuralwavefunction.h"
#include "WaveFunctions/wavefunction.h"

#include "sampler.h"
#include "particle.h"
#include "Math/random.h"

#include <cmath>

#include <iostream>

RmspropGD::RmspropGD(double learningRate,
                     double decayRate,
                     double epsilon,
                     int maxIter,
                     double stepLength,
                     int numberOfMetropolisSteps,
                     int numberOfHiddenNodes,
                     int numberOfDimensions,
                     int numberOfParticles)
    : Optimizer(
          learningRate,
          maxIter,
          stepLength,
          numberOfMetropolisSteps,
          numberOfHiddenNodes,
          numberOfDimensions,
          numberOfParticles)
{
    m_decayRate = decayRate;
    m_epsilon = epsilon;
}

std::unique_ptr<class Sampler> RmspropGD::optimize(
    System &system,
    class NeuralWaveFunction &waveFunction,
    std::vector<std::unique_ptr<class Particle>> &particles,
    std::string filename)
{
    std::cout << "### RMSprop Gradient Descent ###" << std::endl;
    int epoch = 0;

    m_numberOfDimensions = system.getNumberOfDimensions();
    m_numberOfParticles = system.getNumberOfParticles();

    m_msWeights = std::vector<std::vector<std::vector<double>>>(
        m_numberOfParticles,
        std::vector<std::vector<double>>(
            m_numberOfDimensions,
            std::vector<double>(m_numberOfHiddenNodes, 0.0)));

    m_msVisBias = std::vector<std::vector<double>>(
        m_numberOfParticles,
        std::vector<double>(m_numberOfDimensions, 0.0));

    m_msHidBias = std::vector<double>(m_numberOfHiddenNodes, 0.0);

    std::vector<double> hidEnergyDer;
    std::vector<double> gradNorms;
    std::vector<std::vector<double>> visEnergyDer;
    std::vector<std::vector<std::vector<double>>> weightEnergyDer;

    std::vector<double> hiddenBias = waveFunction.getHiddenBias();
    std::vector<std::vector<double>> visibleBias = waveFunction.getVisibleBias();
    std::vector<std::vector<std::vector<double>>> weights = waveFunction.getWeights();

    bool importSamples = system.getImportSamples();
    bool analytical = system.getAnalytical();
    bool interaction = system.getInteraction();

    // declare sampler
    auto sampler = std::make_unique<Sampler>(m_numberOfParticles,
                                             m_numberOfDimensions,
                                             m_numberOfHiddenNodes,
                                             m_stepLength, m_numberOfMetropolisSteps);

    while (epoch < m_maxIter)
    {
        // gradientNorm = 0;
        /*Positions are reset to what they were after equilibration, but the parameters of the wave function should be what they were at the END of last epoch*/
        for (unsigned int i = 0; i < m_numberOfParticles; i++)
        {
            particles[i]->resetEquilibrationPosition();
        }

        sampler = system.runMetropolisSteps(m_stepLength, m_numberOfMetropolisSteps);

        visEnergyDer = sampler->getVisEnergyDer();
        hidEnergyDer = sampler->getHidEnergyDer();
        weightEnergyDer = sampler->getWeightEnergyDer();

        visibleBias = waveFunction.getVisibleBias();
        hiddenBias = waveFunction.getHiddenBias();
        weights = waveFunction.getWeights();

        gradNorms = computeGradientNorms(hidEnergyDer, visEnergyDer, weightEnergyDer);

        sampler->writeGradientSearchToFile(
            system,
            filename,
            epoch,
            gradNorms,
            importSamples,
            analytical,
            interaction,
            m_learningRate);
        // before updating bias and weights, we need to update the ms
        // update visible bias

        for (int i = 0; i < m_numberOfParticles; i++)
        {

            for (int j = 0; j < m_numberOfDimensions; j++)
            {

                m_msVisBias[i][j] = m_decayRate * m_msVisBias[i][j] + (1 - m_decayRate) * visEnergyDer[i][j] * visEnergyDer[i][j];
                visibleBias[i][j] -= m_learningRate * visEnergyDer[i][j] / (std::sqrt(m_msVisBias[i][j]) + m_epsilon);
            }
        }

        // update hidden units
        for (int i = 0; i < m_numberOfHiddenNodes; i++)
        {
            m_msHidBias[i] = m_decayRate * m_msHidBias[i] + (1 - m_decayRate) * hidEnergyDer[i] * hidEnergyDer[i];
            hiddenBias[i] -= m_learningRate * hidEnergyDer[i] / (std::sqrt(m_msHidBias[i]) + m_epsilon);
        }

        // update weights
        for (int i = 0; i < m_numberOfParticles; i++)
        {
            for (int j = 0; j < m_numberOfDimensions; j++)
            {
                for (int k = 0; k < m_numberOfHiddenNodes; k++)
                {
                    m_msWeights[i][j][k] = m_decayRate * m_msWeights[i][j][k] + (1 - m_decayRate) * weightEnergyDer[i][j][k] * weightEnergyDer[i][j][k];
                    weights[i][j][k] -= m_learningRate * weightEnergyDer[i][j][k] / (std::sqrt(m_msWeights[i][j][k]) + m_epsilon);
                }
            }
        }

        epoch++;

        // if we want to stop based on gradient norm critetion, uncomment this
        // gradNorms = computeGradientNorms(hidEnergyDer, visEnergyDer, weightEnergyDer);
        // sampler->writeGradientSearchToFile(system, filename + "_gradient", epoch, gradNorms, m_importSamples, m_analytical, m_interaction, learningRate);

        // set new wave function parameters
        waveFunction.setVisibleBias(visibleBias);
        waveFunction.setHiddenBias(hiddenBias);
        waveFunction.setWeights(weights);

        std::cout << "epoch: " << epoch << "\n";
        std::cout << "energy: " << sampler->getEnergy() << "\n";
    }
    return sampler;
}

std::vector<double> RmspropGD::computeGradientNorms(
    std::vector<double> hidEnergyDer,
    std::vector<std::vector<double>> visEnergyDer,
    std::vector<std::vector<std::vector<double>>> weightEnergyDer)
{
    // output norm of hidEnergyDer
    double hidEnergyDerNorm = 0;
    double visEnergyDerNorm = 0;
    double weightEnergyDerNorm = 0;

    for (int i = 0; i < m_numberOfHiddenNodes; i++)
    {
        hidEnergyDerNorm += hidEnergyDer[i] * hidEnergyDer[i];
    }

    for (int i = 0; i < m_numberOfParticles; i++)
    {
        for (int j = 0; j < m_numberOfDimensions; j++)
        {
            visEnergyDerNorm += visEnergyDer[i][j] * visEnergyDer[i][j];
        }
    }

    // output norm of weightEnergyDer
    for (int i = 0; i < m_numberOfParticles; i++)
    {
        for (int j = 0; j < m_numberOfDimensions; j++)
        {
            for (int k = 0; k < m_numberOfHiddenNodes; k++)
            {
                weightEnergyDerNorm += weightEnergyDer[i][j][k] * weightEnergyDer[i][j][k];
            }
        }
    }

    weightEnergyDerNorm = std::sqrt(weightEnergyDerNorm);
    visEnergyDerNorm = std::sqrt(visEnergyDerNorm);
    hidEnergyDerNorm = std::sqrt(hidEnergyDerNorm);

    return {hidEnergyDerNorm, visEnergyDerNorm, weightEnergyDerNorm};
}