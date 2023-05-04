
#include "optimizer.h"
#include "Math/random.h"

#include <iostream>

Optimizer::Optimizer(
    double learningRate,
    int maxIter,
    double stepLength,
    int numberOfMetropolisSteps,
    int numberOfHiddenNodes,
    int numberOfDimensions,
    int numberOfParticles)
{
    m_learningRate = learningRate;
    m_maxIter = maxIter;
    m_stepLength = stepLength;
    m_numberOfMetropolisSteps = numberOfMetropolisSteps;
    m_numberOfHiddenNodes = numberOfHiddenNodes;
    m_numberOfDimensions = numberOfDimensions;
    m_numberOfParticles = numberOfParticles;
}

std::vector<double> Optimizer::computeGradientNorms(
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

    weightEnergyDerNorm = sqrt(weightEnergyDerNorm);
    visEnergyDerNorm = sqrt(visEnergyDerNorm);
    hidEnergyDerNorm = sqrt(hidEnergyDerNorm);

    return {hidEnergyDerNorm, visEnergyDerNorm, weightEnergyDerNorm};
}