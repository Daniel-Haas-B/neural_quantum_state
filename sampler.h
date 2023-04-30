#pragma once
#include <memory>
#include <string>
#include <fstream>
#include <string>
#include <vector>

class Sampler
{
public:
    Sampler(
        unsigned int numberOfParticles,
        unsigned int numberOfDimensions,
        unsigned int numberOfHiddenNodes,
        double stepLength,
        unsigned int numberOfMetropolisSteps);

    void sample(bool acceptedStep, class System *system);
    void printOutputToTerminal(class System &system);
    void writeOutToFile(class System &system, std::string filename, double omega, bool analytical, bool importanceSampling, bool interaction);
    void WriteTimingToFiles(System &system, std::string filename, bool analytical, unsigned int numberOfEquilibrationSteps, double timing);
    void writeGradientSearchToFile(System &system, std::string filename, int epoch, std::vector<double> gradNormsbool, bool impoSamp, bool analytical, bool interaction, double learningRate);

    void output(System &system, std::string filename, double omega, bool analytical, bool importanceSampling, bool interaction);
    void computeAverages(double cumWeight2, double lambda_l2);

    std::vector<std::vector<double>> getVisEnergyDer();

    std::vector<double> getHidEnergyDer();
    std::vector<std::vector<std::vector<double>>> getWeightEnergyDer();

    double getEnergy()
    {
        return m_energy;
    }

    // Save samples during calculation
    void openSaveSample(std::string filename);
    void saveSample(unsigned int iteration);
    void closeSaveSample();

private:
    double m_stepLength = 0;
    unsigned int m_stepNumber = 0;
    unsigned int m_numberOfMetropolisSteps = 0;
    unsigned int m_numberOfEquilibrationSteps = 0;
    unsigned int m_numberOfParticles = 0;
    unsigned int m_numberOfDimensions = 0;
    unsigned int m_numberOfAcceptedSteps = 0;

    double m_energy = 0;
    double m_energy_variance = 0;
    double m_energy_std = 0;
    double m_acceptRatio = 0;

    double m_cumEnergy = 0;
    double m_cumEnergy2 = 0;

    std::vector<std::vector<std::vector<double>>> m_cumWeightDerPsiE = std::vector<std::vector<std::vector<double>>>(m_numberOfParticles, std::vector<std::vector<double>>(m_numberOfDimensions, std::vector<double>(m_numberOfHiddenNodes, 0)));
    std::vector<std::vector<std::vector<double>>> m_cumWeightDeltaPsi = std::vector<std::vector<std::vector<double>>>(m_numberOfParticles, std::vector<std::vector<double>>(m_numberOfDimensions, std::vector<double>(m_numberOfHiddenNodes, 0)));
    std::vector<std::vector<std::vector<double>>> m_weightDerPsiE = std::vector<std::vector<std::vector<double>>>(m_numberOfParticles, std::vector<std::vector<double>>(m_numberOfDimensions, std::vector<double>(m_numberOfHiddenNodes, 0)));
    std::vector<std::vector<std::vector<double>>> m_weightDeltaPsi = std::vector<std::vector<std::vector<double>>>(m_numberOfParticles, std::vector<std::vector<double>>(m_numberOfDimensions, std::vector<double>(m_numberOfHiddenNodes, 0)));

    std::vector<double> m_cumHidDerPsiE = std::vector<double>(m_numberOfHiddenNodes, 0);
    std::vector<double> m_cumHidDeltaPsi = std::vector<double>(m_numberOfHiddenNodes, 0);
    std::vector<double> m_hidDerPsiE = std::vector<double>(m_numberOfHiddenNodes, 0);
    std::vector<double> m_hidDeltaPsi = std::vector<double>(m_numberOfHiddenNodes, 0);

    std::vector<std::vector<double>> m_cumVisDerPsiE = std::vector<std::vector<double>>(m_numberOfParticles, std::vector<double>(m_numberOfDimensions, 0));
    std::vector<std::vector<double>> m_cumVisDeltaPsi = std::vector<std::vector<double>>(m_numberOfParticles, std::vector<double>(m_numberOfDimensions, 0));
    std::vector<std::vector<double>> m_visDerPsiE = std::vector<std::vector<double>>(m_numberOfParticles, std::vector<double>(m_numberOfDimensions, 0));
    std::vector<std::vector<double>> m_visDeltaPsi = std::vector<std::vector<double>>(m_numberOfParticles, std::vector<double>(m_numberOfDimensions, 0));

    int m_numberOfHiddenNodes = 0;

    std::vector<std::vector<double>> m_visEnergyDer = std::vector<std::vector<double>>(m_numberOfParticles, std::vector<double>(m_numberOfDimensions, 0));
    std::vector<double> m_hidEnergyDer = std::vector<double>(m_numberOfHiddenNodes, 0);
    std::vector<std::vector<std::vector<double>>> m_weightEnergyDer = std::vector<std::vector<std::vector<double>>>(m_numberOfParticles, std::vector<std::vector<double>>(m_numberOfDimensions, std::vector<double>(m_numberOfHiddenNodes, 0)));

    // Save samples during calculation
    std::ofstream m_saveSamplesFile;
};
