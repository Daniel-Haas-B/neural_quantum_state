#include <memory>
#include <vector>

#include "adamGD.h"
#include "WaveFunctions/neuralwavefunction.h"
#include "WaveFunctions/wavefunction.h"

#include "sampler.h"
#include "particle.h"
#include "Math/random.h"

#include <iostream>

AdamGD::AdamGD(double learningRate,
               double beta1,
               double beta2,
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
    m_beta1 = beta1;
    m_beta2 = beta2;
    m_epsilon = epsilon;
}

std::unique_ptr<class Sampler> AdamGD::optimize(
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
    bool analytical = system.getAnalytical();
    bool interaction = system.getInteraction();

    // declare sampler
    auto sampler = std::make_unique<Sampler>(m_numberOfParticles,
                                             m_numberOfDimensions,
                                             m_numberOfHiddenNodes,
                                             m_stepLength, m_numberOfMetropolisSteps);

    // now we use ADAM
    std::vector<std::vector<std::vector<double>>> m_t_weights(weights.size(), std::vector<std::vector<double>>(weights[0].size(), std::vector<double>(weights[0][0].size(), 0)));
    std::vector<std::vector<double>> m_t_vis(visibleBias.size(), std::vector<double>(visibleBias[0].size(), 0));
    std::vector<double> m_t_hid(hiddenBias.size(), 0);

    std::vector<std::vector<std::vector<double>>> s_t_weights(weights.size(), std::vector<std::vector<double>>(weights[0].size(), std::vector<double>(weights[0][0].size(), 0)));
    std::vector<std::vector<double>> s_t_vis(visibleBias.size(), std::vector<double>(visibleBias[0].size(), 0));
    std::vector<double> s_t_hid(hiddenBias.size(), 0);

    std::vector<std::vector<std::vector<double>>> m_t_hat_weights(weights.size(), std::vector<std::vector<double>>(weights[0].size(), std::vector<double>(weights[0][0].size(), 0)));
    std::vector<std::vector<double>> m_t_hat_vis(visibleBias.size(), std::vector<double>(visibleBias[0].size(), 0));
    std::vector<double> m_t_hat_hid(hiddenBias.size(), 0);

    std::vector<std::vector<std::vector<double>>> s_t_hat_weights(weights.size(), std::vector<std::vector<double>>(weights[0].size(), std::vector<double>(weights[0][0].size(), 0)));
    std::vector<std::vector<double>> s_t_hat_vis(visibleBias.size(), std::vector<double>(visibleBias[0].size(), 0));
    std::vector<double> s_t_hat_hid(hiddenBias.size(), 0);

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

        unsigned int i = 0;
        unsigned int j = 0;
        unsigned int k = 0;

        for (i = 0; i < m_numberOfParticles; i++)
        {
            for (j = 0; j < m_numberOfDimensions; j++)
            {
                m_t_vis[i][j] = m_beta1 * m_t_vis[i][j] + (1 - m_beta1) * visEnergyDer[i][j];
                s_t_vis[i][j] = m_beta2 * s_t_vis[i][j] + (1 - m_beta2) * visEnergyDer[i][j] * visEnergyDer[i][j];

                m_t_hat_vis[i][j] = m_t_vis[i][j] / (1 - pow(m_beta1, epoch + 1));
                s_t_hat_vis[i][j] = s_t_vis[i][j] / (1 - pow(m_beta2, epoch + 1));

                visibleBias[i][j] -= m_learningRate * m_t_hat_vis[i][j] / (sqrt(s_t_hat_vis[i][j]) + m_epsilon);
            }
        }

        // update hidden units
        for (i = 0; i < m_numberOfHiddenNodes; i++)
        {
            m_t_hid[i] = m_beta1 * m_t_hid[i] + (1 - m_beta1) * hidEnergyDer[i];
            s_t_hid[i] = m_beta2 * s_t_hid[i] + (1 - m_beta2) * hidEnergyDer[i] * hidEnergyDer[i];

            m_t_hat_hid[i] = m_t_hid[i] / (1 - pow(m_beta1, epoch + 1));
            s_t_hat_hid[i] = s_t_hid[i] / (1 - pow(m_beta2, epoch + 1));

            hiddenBias[i] -= m_learningRate * m_t_hat_hid[i] / (sqrt(s_t_hat_hid[i]) + m_epsilon);
        }

        // update weights

        double w_decay;

        // double l2_lambda = 0.0001;
        for (i = 0; i < m_numberOfParticles; i++)
        {
            for (j = 0; j < m_numberOfDimensions; j++)
            {
                for (k = 0; k < m_numberOfHiddenNodes; k++)
                {
                    m_t_weights[i][j][k] = m_beta1 * m_t_weights[i][j][k] + (1 - m_beta1) * weightEnergyDer[i][j][k];
                    s_t_weights[i][j][k] = m_beta2 * s_t_weights[i][j][k] + (1 - m_beta2) * weightEnergyDer[i][j][k] * weightEnergyDer[i][j][k];

                    m_t_hat_weights[i][j][k] = m_t_weights[i][j][k] / (1 - pow(m_beta1, epoch + 1));
                    s_t_hat_weights[i][j][k] = s_t_weights[i][j][k] / (1 - pow(m_beta2, epoch + 1));

                    // L2 REGULARIZATION

                    w_decay = 0; // l2_lambda * weights[i][j][k];

                    weights[i][j][k] -= m_learningRate * (m_t_hat_weights[i][j][k] / (sqrt(s_t_hat_weights[i][j][k]) + m_epsilon) + w_decay);
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

        // m_learningRate *= 0.99;
    }
    return sampler;
}
