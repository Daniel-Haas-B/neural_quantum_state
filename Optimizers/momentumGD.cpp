#include <memory>
#include <vector>

#include "momentumGD.h"
#include "WaveFunctions/neuralwavefunction.h"
#include "WaveFunctions/wavefunction.h"

#include "sampler.h"
#include "particle.h"
#include "Math/random.h"

#include <iostream>

MomentumGD::MomentumGD(double learningRate,
                       double gamma,
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
    m_gamma = gamma;
}

std::unique_ptr<class Sampler> MomentumGD::optimize(
    System &system,
    class NeuralWaveFunction &waveFunction,
    std::vector<std::unique_ptr<class Particle>> &particles,
    std::string filename)
{

    unsigned int epoch = 0;

    std::vector<double> hidEnergyDer;
    std::vector<double> gradNorms;
    std::vector<std::vector<double>> visEnergyDer;
    std::vector<std::vector<std::vector<double>>> weightEnergyDer;

    std::vector<double> hiddenBias = waveFunction.getHiddenBias();
    std::vector<std::vector<double>> visibleBias = waveFunction.getVisibleBias();
    std::vector<std::vector<std::vector<double>>> weights = waveFunction.getWeights();

    bool importSamples = system.getImportSamples();
    std::string optimizerType = system.getOptimizerType();
    bool interaction = system.getInteraction();

    // declare sampler
    auto sampler = std::make_unique<Sampler>(m_numberOfParticles,
                                             m_numberOfDimensions,
                                             m_numberOfHiddenNodes,
                                             m_stepLength, m_numberOfMetropolisSteps);

    // now we use momentum so we need to keep track of the previous step, initialize to zero
    std::vector<double> velocityHid(m_numberOfHiddenNodes, 0);
    std::vector<std::vector<double>> velocityVis(m_numberOfParticles, std::vector<double>(m_numberOfDimensions, 0));
    std::vector<std::vector<std::vector<double>>> velocityWeights(m_numberOfHiddenNodes, std::vector<std::vector<double>>(m_numberOfParticles, std::vector<double>(m_numberOfDimensions, 0)));

    while (epoch < m_maxIter)
    {
        /*Positions are reset to what they were after equilibration, but the parameters of the wave function should be what they were at the END of last epoch*/
        unsigned int i, j, k;
        for (i = 0; i < m_numberOfParticles; i++)
        {
            particles[i]->resetEquilibrationPosition();
        }

        sampler->reset();
        system.runMetropolisSteps(sampler, m_stepLength, m_numberOfMetropolisSteps);

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
            optimizerType,
            interaction,
            m_learningRate);

        for (i = 0; i < m_numberOfParticles; i++)
        {
            for (j = 0; j < m_numberOfDimensions; j++)
            {
                velocityVis[i][j] = m_gamma * velocityVis[i][j] + m_learningRate * visEnergyDer[i][j];
                visibleBias[i][j] -= velocityVis[i][j];
            }
        }

        // update hidden units
        for (i = 0; i < m_numberOfHiddenNodes; i++)
        {
            velocityHid[i] = m_gamma * velocityHid[i] + m_learningRate * hidEnergyDer[i];
            hiddenBias[i] -= velocityHid[i];
        }

        // update weights
        for (i = 0; i < m_numberOfParticles; i++)
        {
            for (j = 0; j < m_numberOfDimensions; j++)
            {
                for (k = 0; k < m_numberOfHiddenNodes; k++)
                {
                    velocityWeights[i][j][k] = m_gamma * velocityWeights[i][j][k] + m_learningRate * weightEnergyDer[i][j][k];
                    weights[i][j][k] -= velocityWeights[i][j][k];
                }
            }
        }

        epoch++;
        // set new wave function parameters
        waveFunction.setVisibleBias(visibleBias);
        waveFunction.setHiddenBias(hiddenBias);
        waveFunction.setWeights(weights);

        if (epoch % 10 == 0)
        {
            std::cout << "epoch: " << epoch << "\n";
            std::cout << "energy: " << sampler->getEnergy() << "\n";
        }
    }
    return sampler;
}
