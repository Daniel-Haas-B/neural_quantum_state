
#include "optimizer.h"
#include "Math/random.h"

#include <iostream>

Optimizer::Optimizer(
    double learningRate,
    int maxIter,
    double stepLength,
    int numberOfMetropolisSteps,
    int numberOfHiddenNodes)
{
    m_learningRate = learningRate;
    m_maxIter = maxIter;
    m_stepLength = stepLength;
    m_numberOfMetropolisSteps = numberOfMetropolisSteps;
    m_numberOfHiddenNodes = numberOfHiddenNodes;
}
