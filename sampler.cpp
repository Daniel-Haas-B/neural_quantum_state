#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include "system.h"
#include "sampler.h"
#include "particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/wavefunction.h"

using std::cout;
using std::endl;

using std::fixed;
using std::setprecision;
using std::setw;

Sampler::Sampler(
    unsigned int numberOfParticles,
    unsigned int numberOfDimensions,
    unsigned int numberOfHiddenNodes,
    double stepLength,
    unsigned int numberOfMetropolisSteps)
{
    m_stepNumber = 0;
    m_numberOfMetropolisSteps = numberOfMetropolisSteps;
    m_numberOfParticles = numberOfParticles;
    m_numberOfDimensions = numberOfDimensions;
    m_numberOfHiddenNodes = numberOfHiddenNodes;
    m_stepLength = stepLength;

    m_energy = 0;
    m_energy_variance = 0;
    m_energy_std = 0;

    m_cumEnergy = 0;
    m_cumEnergy2 = 0;
    m_numberOfAcceptedSteps = 0;

    m_cumHidDerPsiE = std::vector<double>(numberOfHiddenNodes, 0);
    m_cumVisDerPsiE = std::vector<std::vector<double>>(numberOfParticles, std::vector<double>(numberOfDimensions, 0));
    m_cumWeightDerPsiE = std::vector<std::vector<std::vector<double>>>(numberOfParticles, std::vector<std::vector<double>>(numberOfDimensions, std::vector<double>(numberOfHiddenNodes, 0)));

    m_cumHidDeltaPsi = std::vector<double>(numberOfHiddenNodes, 0);
    m_cumVisDeltaPsi = std::vector<std::vector<double>>(numberOfParticles, std::vector<double>(numberOfDimensions, 0));
    m_cumWeightDeltaPsi = std::vector<std::vector<std::vector<double>>>(numberOfParticles, std::vector<std::vector<double>>(numberOfDimensions, std::vector<double>(numberOfHiddenNodes, 0)));

    m_hidDeltaPsi = std::vector<double>(numberOfHiddenNodes, 0);
    m_visDeltaPsi = std::vector<std::vector<double>>(numberOfParticles, std::vector<double>(numberOfDimensions, 0));
    m_weightDeltaPsi = std::vector<std::vector<std::vector<double>>>(numberOfParticles, std::vector<std::vector<double>>(numberOfDimensions, std::vector<double>(numberOfHiddenNodes, 0)));

    m_hidDerPsiE = std::vector<double>(numberOfHiddenNodes, 0);
    m_visDerPsiE = std::vector<std::vector<double>>(numberOfParticles, std::vector<double>(numberOfDimensions, 0));
    m_weightDerPsiE = std::vector<std::vector<std::vector<double>>>(numberOfParticles, std::vector<std::vector<double>>(numberOfDimensions, std::vector<double>(numberOfHiddenNodes, 0)));

    m_weightEnergyDer = std::vector<std::vector<std::vector<double>>>(numberOfParticles, std::vector<std::vector<double>>(numberOfDimensions, std::vector<double>(numberOfHiddenNodes, 0)));
    m_visEnergyDer = std::vector<std::vector<double>>(numberOfParticles, std::vector<double>(numberOfDimensions, 0));
    m_hidEnergyDer = std::vector<double>(numberOfHiddenNodes, 0);
}

void Sampler::sample(bool acceptedStep, System *system)
{
    /* Here you should sample all the interesting things you want to measure.
     * Note that there are (way) more than the single one here currently.
     */

    auto localEnergy = system->computeLocalEnergy();

    m_cumEnergy += localEnergy;
    m_cumEnergy2 += (localEnergy * localEnergy);
    m_stepNumber++;
    m_numberOfAcceptedSteps += acceptedStep;

    system->computeParamDerivative(m_weightDeltaPsi,
                                   m_visDeltaPsi,
                                   m_hidDeltaPsi);

    unsigned int i = 0;
    unsigned int j = 0;
    unsigned int k = 0;

    for (i = 0; i < m_numberOfHiddenNodes; i++)
    {
        m_hidDerPsiE[i] = m_hidDeltaPsi[i] * localEnergy;
        m_cumHidDeltaPsi[i] += m_hidDeltaPsi[i];
        m_cumHidDerPsiE[i] += m_hidDerPsiE[i];
    }

    for (i = 0; i < m_numberOfParticles; i++)
    {
        for (j = 0; j < m_numberOfDimensions; j++)
        {
            m_visDerPsiE[i][j] = m_visDeltaPsi[i][j] * localEnergy;
            m_cumVisDeltaPsi[i][j] += m_visDeltaPsi[i][j];
            m_cumVisDerPsiE[i][j] += m_visDerPsiE[i][j];

            for (k = 0; k < m_numberOfHiddenNodes; k++)
            {
                m_weightDerPsiE[i][j][k] = m_weightDeltaPsi[i][j][k] * localEnergy;
                m_cumWeightDeltaPsi[i][j][k] += m_weightDeltaPsi[i][j][k];
                m_cumWeightDerPsiE[i][j][k] += m_weightDerPsiE[i][j][k];
            }
        }
    }
}

void Sampler::computeAverages()
{
    /* Compute the averages of the sampled quantities.
     */
    m_energy = m_cumEnergy / m_numberOfMetropolisSteps; // + (lambda_l2 / 2) * cumWeight2;
    m_cumEnergy2 /= m_numberOfMetropolisSteps;
    m_energy_variance = (m_cumEnergy2 - m_energy * m_energy);
    m_energy_std = sqrt(m_energy_variance);
    m_acceptRatio = ((double)m_numberOfAcceptedSteps) / ((double)m_numberOfMetropolisSteps);

    // compute averages for hidden
    for (int i = 0; i < m_numberOfHiddenNodes; i++)
    {
        m_hidDerPsiE[i] = m_cumHidDerPsiE[i] / m_numberOfMetropolisSteps;
        m_hidDeltaPsi[i] = m_cumHidDeltaPsi[i] / m_numberOfMetropolisSteps;
    }

    // compute averages for weights
    for (int i = 0; i < m_numberOfParticles; i++)
    {
        for (int j = 0; j < m_numberOfDimensions; j++)
        {
            m_visDerPsiE[i][j] = m_cumVisDerPsiE[i][j] / m_numberOfMetropolisSteps;
            m_visDeltaPsi[i][j] = m_cumVisDeltaPsi[i][j] / m_numberOfMetropolisSteps;
            for (int k = 0; k < m_numberOfHiddenNodes; k++)
            {
                m_weightDerPsiE[i][j][k] = m_cumWeightDerPsiE[i][j][k] / m_numberOfMetropolisSteps;
                m_weightDeltaPsi[i][j][k] = m_cumWeightDeltaPsi[i][j][k] / m_numberOfMetropolisSteps;
            }
        }
    }
}

std::vector<std::vector<double>> &Sampler::getVisEnergyDer()
{
    for (int i = 0; i < m_numberOfParticles; i++) // number of particles
    {
        for (int j = 0; j < m_numberOfDimensions; j++)
        {
            m_visEnergyDer[i][j] = 2 * (m_visDerPsiE[i][j] - m_visDeltaPsi[i][j] * m_energy);
        }
    }
    return m_visEnergyDer;
}

std::vector<double> &Sampler::getHidEnergyDer()
{
    for (int i = 0; i < m_numberOfHiddenNodes; i++) // number of hidden nodes
    {
        m_hidEnergyDer[i] = 2 * (m_hidDerPsiE[i] - m_hidDeltaPsi[i] * m_energy);
    }
    return m_hidEnergyDer;
}

std::vector<std::vector<std::vector<double>>> &Sampler::getWeightEnergyDer()
{
    for (int i = 0; i < m_numberOfParticles; i++)
    {
        for (int j = 0; j < m_numberOfDimensions; j++)
        {
            for (int k = 0; k < m_numberOfHiddenNodes; k++)
            {
                m_weightEnergyDer[i][j][k] = 2 * (m_weightDerPsiE[i][j][k] - m_weightDeltaPsi[i][j][k] * m_energy);
            }
        }
    }
    return m_weightEnergyDer;
}

void Sampler::reset()
{
    /*Cleans all the measusements*/
    m_stepNumber = 0;
    m_numberOfAcceptedSteps = 0;
    m_energy = 0;
    m_energy_variance = 0;
    m_energy_std = 0;
    m_acceptRatio = 0;
    m_cumEnergy = 0;
    m_cumEnergy2 = 0;

    m_visDeltaPsi = std::vector<std::vector<double>>(m_numberOfParticles, std::vector<double>(m_numberOfDimensions, 0));
    m_weightDeltaPsi = std::vector<std::vector<std::vector<double>>>(m_numberOfParticles, std::vector<std::vector<double>>(m_numberOfDimensions, std::vector<double>(m_numberOfHiddenNodes, 0)));
    m_hidDeltaPsi = std::vector<double>(m_numberOfHiddenNodes, 0);

    m_cumWeightDerPsiE = std::vector<std::vector<std::vector<double>>>(m_numberOfParticles, std::vector<std::vector<double>>(m_numberOfDimensions, std::vector<double>(m_numberOfHiddenNodes, 0)));
    m_cumWeightDeltaPsi = std::vector<std::vector<std::vector<double>>>(m_numberOfParticles, std::vector<std::vector<double>>(m_numberOfDimensions, std::vector<double>(m_numberOfHiddenNodes, 0)));

    m_cumVisDerPsiE = std::vector<std::vector<double>>(m_numberOfParticles, std::vector<double>(m_numberOfDimensions, 0));
    m_cumVisDeltaPsi = std::vector<std::vector<double>>(m_numberOfParticles, std::vector<double>(m_numberOfDimensions, 0));

    m_cumHidDerPsiE = std::vector<double>(m_numberOfHiddenNodes, 0);
    m_cumHidDeltaPsi = std::vector<double>(m_numberOfHiddenNodes, 0);
}

void Sampler::printOutputToTerminal(System &system)
{
    cout << endl;
    cout << "  -- System info -- " << endl;
    cout << " Number of particles  : " << m_numberOfParticles << endl;
    cout << " Number of dimensions : " << m_numberOfDimensions << endl;
    cout << " Number of Metropolis steps run : 10^" << std::log10(m_numberOfMetropolisSteps) << " = 2^" << std::log2(m_numberOfMetropolisSteps) << endl;
    cout << " Step length used : " << m_stepLength << endl;
    cout << " Ratio of accepted steps: " << ((double)m_numberOfAcceptedSteps) / ((double)m_numberOfMetropolisSteps) << endl;
    cout << endl;
    cout << "  -- Wave function parameters -- " << endl;
    cout << "  -- Results -- " << endl;
    cout << " Energy : " << m_energy << endl;
    cout << " Energy variance : " << m_energy_variance << endl;
    cout << " Energy std : " << m_energy_std << endl;
    cout << endl;
}

void Sampler::writeOutToFile(System &system, std::string filename, double omega, std::string optimizerType, bool importanceSampling, bool interaction)
{
    std::ifstream exsists_file(filename.c_str());

    std::fstream outfile;
    int w = 20;

    if (!exsists_file.good())
    {
        outfile.open(filename, std::ios::out);
        outfile << setw(w) << "Dimensions"
                << setw(w) << "Particles"
                << setw(w) << "Hidden-nodes"
                << setw(w) << "Metro-steps"
                << setw(w) << "Omega"
                << setw(w) << "StepLength";
        outfile << setw(w) << "Energy"
                << setw(w) << "Energy_std"
                << setw(w) << "Energy_var"
                << setw(w) << "Accept_number"
                << setw(w) << "Accept_ratio"
                << setw(w) << "Imposampling"
                << setw(w) << "optimizerType"
                << setw(w) << "Interaction"
                << "\n";
    }
    else
    {
        outfile.open(filename, std::ios::out | std::ios::app);
    }
    outfile << setw(w) << m_numberOfDimensions
            << setw(w) << m_numberOfParticles
            << setw(w) << m_numberOfHiddenNodes
            << setw(w) << setprecision(8) << m_numberOfMetropolisSteps
            << setw(w) << fixed << setprecision(8) << omega
            << setw(w) << fixed << setprecision(8) << m_stepLength;

    outfile << setw(w) << fixed << setprecision(8) << m_energy
            << setw(w) << fixed << setprecision(8) << m_energy_std
            << setw(w) << fixed << setprecision(8) << m_energy_variance
            << setw(w) << fixed << setprecision(8) << m_numberOfAcceptedSteps
            << setw(w) << fixed << setprecision(8) << m_acceptRatio
            << setw(w) << fixed << setprecision(8) << importanceSampling
            << setw(w) << fixed << setprecision(8) << optimizerType
            << setw(w) << fixed << setprecision(8) << interaction
            << "\n";

    outfile.close();
}

void Sampler::output(System &system, std::string filename, double omega, std::string optimizerType, bool importanceSampling, bool interaction)
{
    // Output information from the simulation, either as file or print
    if (filename == ".txt") // this is dumbly duplicated now
    {
        printOutputToTerminal(system);
    }
    else
    {
        writeOutToFile(system, filename, omega, optimizerType, importanceSampling, interaction);
    }
}

void Sampler::writeGradientSearchToFile(System &system, std::string filename, int epoch, std::vector<double> gradNorms, bool impoSamp, std::string optimizerType, bool interaction, double learningRate)
{ // write out the gradient search to file
    // break filename to add "detailed_" to the beginning, after the path
    std::string path = filename.substr(0, filename.find_last_of("/\\") + 1);
    std::string filename_only = filename.substr(filename.find_last_of("/\\") + 1);
    filename = path + "detailed_" + filename_only + ".txt";
    std::ifstream exsists_file(filename.c_str());

    std::fstream outfile;

    int w = 20;

    if (!exsists_file.good())
    {
        outfile.open(filename, std::ios::out);
        outfile << setw(w) << "Dimensions"
                << setw(w) << "Particles"
                << setw(w) << "Hidden-nodes"
                << setw(w) << "Metro-steps"
                << setw(w) << "StepLength";
        outfile << setw(w) << "Energy"
                << setw(w) << "Energy_std"
                << setw(w) << "Energy_var"
                << setw(w) << "Accept_number"
                << setw(w) << "Accept_ratio"
                << setw(w) << "Imposampling"
                << setw(w) << "optimizerType"
                << setw(w) << "Interaction"
                << setw(w) << "Epoch"
                << setw(w) << "LearningRate"
                << setw(w) << "HiddenGradNorm"
                << setw(w) << "VisibleGradNorm"
                << setw(w) << "WeightGradNorm"
                << "\n";
    }
    else
    {
        outfile.open(filename, std::ios::out | std::ios::app);
    }
    outfile << setw(w) << m_numberOfDimensions
            << setw(w) << m_numberOfParticles
            << setw(w) << m_numberOfHiddenNodes
            << setw(w) << setprecision(8) << m_numberOfMetropolisSteps
            << setw(w) << fixed << setprecision(8) << m_stepLength
            << setw(w) << fixed << setprecision(8) << m_energy
            << setw(w) << fixed << setprecision(8) << m_energy_std
            << setw(w) << fixed << setprecision(8) << m_energy_variance
            << setw(w) << fixed << setprecision(8) << m_numberOfAcceptedSteps
            << setw(w) << fixed << setprecision(8) << m_acceptRatio
            << setw(w) << fixed << setprecision(8) << impoSamp
            << setw(w) << fixed << setprecision(8) << optimizerType
            << setw(w) << fixed << setprecision(8) << interaction
            << setw(w) << fixed << setprecision(8) << epoch
            << setw(w) << fixed << setprecision(8) << learningRate
            << setw(w) << fixed << setprecision(8) << gradNorms.at(0)
            << setw(w) << fixed << setprecision(8) << gradNorms.at(1)
            << setw(w) << fixed << setprecision(8) << gradNorms.at(2)
            << "\n";

    outfile.close();
}

void Sampler::openSaveSample(std::string filename)
{
    m_saveSamplesFile = std::ofstream(filename, std::ios::out | std::ios::binary | std::ios::trunc); // create binary file
    if (!m_saveSamplesFile)
    {
        std::cerr << "Error: could not open file " << filename << " to store samples." << std::endl;
        exit(1);
    }
}
void Sampler::saveSample(unsigned int iteration)
{
    double energy = m_cumEnergy / double(iteration + 1);
    m_saveSamplesFile.write(reinterpret_cast<const char *>(&energy), sizeof(double));
}
void Sampler::closeSaveSample()
{
    m_saveSamplesFile.close();
}