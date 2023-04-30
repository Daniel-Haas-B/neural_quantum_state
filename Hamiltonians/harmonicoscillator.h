#pragma once
#include <memory>
#include <vector>

#include "hamiltonian.h"

class HarmonicOscillator : public Hamiltonian
{
public:
    HarmonicOscillator(double omega, bool interaction);
    double computeLocalEnergy(
        class NeuralWaveFunction &waveFunction,
        std::vector<std::unique_ptr<class Particle>> &particles);

private:
    double m_omega;
    bool m_interaction;
};
