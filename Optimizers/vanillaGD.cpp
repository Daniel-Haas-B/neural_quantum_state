#include <memory>
#include <vector>

#include "vanillaGD.h"
#include "WaveFunctions/neuralwavefunction.h"
#include "WaveFunctions/wavefunction.h"

#include "sampler.h"
#include "particle.h"
#include "Math/random.h"

#include <iostream>

VanillaGD::VanillaGD(double learningRate,
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
}

std::unique_ptr<class Sampler> VanillaGD::optimize(
    System &system,
    class NeuralWaveFunction &waveFunction,
    std::vector<std::unique_ptr<class Particle>> &particles,
    std::string filename)
{

    int epoch = 0;

    m_numberOfDimensions = system.getNumberOfDimensions();
    m_numberOfParticles = system.getNumberOfParticles();

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

        for (int i = 0; i < m_numberOfParticles; i++)
        {
            for (int j = 0; j < m_numberOfDimensions; j++)
            {
                visibleBias[i][j] -= m_learningRate * visEnergyDer[i][j];
            }
        }

        // update hidden units
        for (int i = 0; i < m_numberOfHiddenNodes; i++)
        {
            hiddenBias[i] -= m_learningRate * hidEnergyDer[i];
        }

        // update weights
        for (int i = 0; i < m_numberOfParticles; i++)
        {
            for (int j = 0; j < m_numberOfDimensions; j++)
            {
                for (int k = 0; k < m_numberOfHiddenNodes; k++)
                {
                    weights[i][j][k] -= m_learningRate * weightEnergyDer[i][j][k];
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
