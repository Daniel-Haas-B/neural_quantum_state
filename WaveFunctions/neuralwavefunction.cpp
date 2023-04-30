#include "neuralwavefunction.h"
#include "Math/random.h"

NeuralWaveFunction::NeuralWaveFunction(std::unique_ptr<class Random> rng)
{
    m_rng = std::move(rng);
}
