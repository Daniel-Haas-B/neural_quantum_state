#pragma once

#include <memory>
#include <vector>
#include <string>

class System
{
public:
    System(std::unique_ptr<class Hamiltonian> hamiltonian,
           std::unique_ptr<class NeuralWaveFunction> waveFunction,
           std::unique_ptr<class MonteCarlo> solver,
           std::vector<std::unique_ptr<class Particle>> particles,
           bool importSamples,
           std::string optimizerType,
           bool interaction);

    unsigned int runEquilibrationSteps(double stepLength,
                                       unsigned int numberOfEquilibrationSteps);

    void runMetropolisSteps(
        std::unique_ptr<class Sampler> &sampler,
        double stepLength,
        unsigned int numberOfMetropolisSteps);

    std::unique_ptr<class Sampler> optimizeMetropolis(
        System &system,
        std::string filename,
        double stepLength,
        unsigned int numberOfMetropolisSteps,
        unsigned int numberOfEquilibrationSteps,
        double beta1,
        double beta2,
        double epsilon,
        double learningRate,
        std::string optimizerType);

    double computeLocalEnergy();

    void computeParamDerivative(std::vector<std::vector<std::vector<double>>> &weightDeltaPsi,
                                std::vector<std::vector<double>> &visDeltaPsi,
                                std::vector<double> &hidDeltaPsi);

    void setWaveFunction(std::unique_ptr<class NeuralWaveFunction> waveFunction);
    void setSolver(std::unique_ptr<class MonteCarlo> solver);

    void saveSamples(std::string filename, int skip);
    int getSkip();
    void saveFinalState(std::string filename);

    bool &getImportSamples() { return m_importSamples; }
    std::string &getOptimizerType() { return m_optimizerType; }
    bool &getInteraction() { return m_interaction; }

    // Getters
    unsigned int getNumberOfParticles() { return m_numberOfParticles; }
    unsigned int getNumberOfDimensions() { return m_numberOfDimensions; }

protected:
    bool m_interaction = false;

private:
    unsigned int m_numberOfParticles = 0;
    unsigned int m_numberOfDimensions = 0;
    unsigned int m_numberOfHiddenNodes = 0;

    bool m_importSamples = false;
    std::string m_optimizerType = "vanillaGD";

    std::unique_ptr<class Hamiltonian> m_hamiltonian;
    std::unique_ptr<class NeuralWaveFunction> m_waveFunction;
    std::unique_ptr<class MonteCarlo> m_solver;
    std::vector<std::unique_ptr<class Particle>> m_particles;

    bool m_saveSamples = false;
    int m_skip = 0;
    std::string m_saveSamplesFilename;
};
