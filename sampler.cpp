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

    m_cumHidDerPsiE = std::vector<double>(m_numberOfHiddenNodes, 0);
    m_cumVisDerPsiE = std::vector<std::vector<double>>(numberOfParticles, std::vector<double>(numberOfDimensions, 0));
    m_cumWeightDerPsiE = std::vector<std::vector<std::vector<double>>>(numberOfParticles, std::vector<std::vector<double>>(numberOfDimensions, std::vector<double>(m_numberOfHiddenNodes, 0)));

    m_cumHidDeltaPsi = std::vector<double>(m_numberOfHiddenNodes, 0);
    m_cumVisDeltaPsi = std::vector<std::vector<double>>(numberOfParticles, std::vector<double>(numberOfDimensions, 0));
    m_cumWeightDeltaPsi = std::vector<std::vector<std::vector<double>>>(numberOfParticles, std::vector<std::vector<double>>(numberOfDimensions, std::vector<double>(m_numberOfHiddenNodes, 0)));

    m_hidDeltaPsi = std::vector<double>(m_numberOfHiddenNodes, 0);
    m_visDeltaPsi = std::vector<std::vector<double>>(numberOfParticles, std::vector<double>(numberOfDimensions, 0));
    m_weightDeltaPsi = std::vector<std::vector<std::vector<double>>>(numberOfParticles, std::vector<std::vector<double>>(numberOfDimensions, std::vector<double>(m_numberOfHiddenNodes, 0)));

    m_hidDerPsiE = std::vector<double>(m_numberOfHiddenNodes, 0);
    m_visDerPsiE = std::vector<std::vector<double>>(numberOfParticles, std::vector<double>(numberOfDimensions, 0));
    m_weightDerPsiE = std::vector<std::vector<std::vector<double>>>(numberOfParticles, std::vector<std::vector<double>>(numberOfDimensions, std::vector<double>(m_numberOfHiddenNodes, 0)));

    m_weightEnergyDer = std::vector<std::vector<std::vector<double>>>(numberOfParticles, std::vector<std::vector<double>>(numberOfDimensions, std::vector<double>(m_numberOfHiddenNodes, 0)));
    m_visEnergyDer = std::vector<std::vector<double>>(numberOfParticles, std::vector<double>(numberOfDimensions, 0));
    m_hidEnergyDer = std::vector<double>(m_numberOfHiddenNodes, 0);
}

void Sampler::sample(bool acceptedStep, System *system)
{
    /* Here you should sample all the interesting things you want to measure.
     * Note that there are (way) more than the single one here currently.
     */

    // std::cout << "DEBUG INSIDE SAMPLER " << std::endl;

    auto localEnergy = system->computeLocalEnergy();

    m_cumEnergy += localEnergy;
    m_cumEnergy2 += (localEnergy * localEnergy);
    m_stepNumber++;
    m_numberOfAcceptedSteps += acceptedStep;

    // deltas and cumulatives for hidden

    system->computeParamDerivative(m_weightDeltaPsi,
                                   m_visDeltaPsi,
                                   m_hidDeltaPsi);

    // m_hidDeltaPsi = system->computeHidBiasDerivative();
    // m_visDeltaPsi = system->computeVisBiasDerivative();
    // m_weightDeltaPsi = system->computeWeightDerivative();

    // deltas and cumulatives for weights

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

void Sampler::printOutputToTerminal(System &system)
{
    // auto pa = system.getWaveFunctionParameters();
    // auto p = pa.size();

    cout << endl;
    cout << "  -- System info -- " << endl;
    cout << " Number of particles  : " << m_numberOfParticles << endl;
    cout << " Number of dimensions : " << m_numberOfDimensions << endl;
    cout << " Number of Metropolis steps run : 10^" << std::log10(m_numberOfMetropolisSteps) << " = 2^" << std::log2(m_numberOfMetropolisSteps) << endl;
    cout << " Step length used : " << m_stepLength << endl;
    cout << " Ratio of accepted steps: " << ((double)m_numberOfAcceptedSteps) / ((double)m_numberOfMetropolisSteps) << endl;
    cout << endl;
    cout << "  -- Wave function parameters -- " << endl;
    // cout << " Number of parameters : " << p << endl;
    // for (unsigned int i = 0; i < p; i++)
    //{
    //     cout << " Parameter " << i + 1 << " : " << pa.at(i) << endl;
    // }
    // cout << endl;
    cout << "  -- Results -- " << endl;
    cout << " Energy : " << m_energy << endl;
    cout << " Energy variance : " << m_energy_variance << endl;
    cout << " Energy std : " << m_energy_std << endl;
    cout << endl;
}

void Sampler::writeOutToFile(System &system, std::string filename, double omega, bool analytical, bool importanceSampling, bool interaction)
{
    std::ifstream exsists_file(filename.c_str());

    std::fstream outfile;
    // auto pa = system.getWaveFunctionParameters();
    // int p = pa.size();
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
        // for (int i = 0; i < p; i++)
        //     outfile << setw(w - 1) << "WF" << (i + 1);

        outfile << setw(w) << "Energy"
                << setw(w) << "Energy_std"
                << setw(w) << "Energy_var"
                << setw(w) << "Accept_number"
                << setw(w) << "Accept_ratio"
                << setw(w) << "Imposampling"
                << setw(w) << "Analytical"
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
            << setw(w) << setprecision(5) << m_numberOfMetropolisSteps
            << setw(w) << fixed << setprecision(5) << omega
            << setw(w) << fixed << setprecision(5) << m_stepLength;

    // for (int i = 0; i < p; i++)
    //     outfile << setw(w) << fixed << setprecision(5) << pa.at(i);

    outfile << setw(w) << fixed << setprecision(5) << m_energy
            << setw(w) << fixed << setprecision(5) << m_energy_std
            << setw(w) << fixed << setprecision(5) << m_energy_variance
            << setw(w) << fixed << setprecision(5) << m_numberOfAcceptedSteps
            << setw(w) << fixed << setprecision(5) << m_acceptRatio
            << setw(w) << fixed << setprecision(5) << importanceSampling
            << setw(w) << fixed << setprecision(5) << analytical
            << setw(w) << fixed << setprecision(5) << interaction
            << "\n";

    outfile.close();
}

void Sampler::output(System &system, std::string filename, double omega, bool analytical, bool importanceSampling, bool interaction)
{
    // Output information from the simulation, either as file or print
    if (filename == ".txt") // this is dumbly duplicated now
    {
        printOutputToTerminal(system);
    }
    else
    {
        writeOutToFile(system, filename, omega, analytical, importanceSampling, interaction);
    }
}

void Sampler::WriteTimingToFiles(System &system, std::string filename, bool analytical, unsigned int numberOfEquilibrationSteps, double timing)
{
    std::ifstream exsists_file(filename.c_str());

    std::fstream outfile;
    // auto pa = system.getWaveFunctionParameters();
    int w = 20;

    if (!exsists_file.good())
    {

        outfile.open(filename, std::ios::out);
        outfile << setw(w) << "Dimensions"
                << setw(w) << "Particles"
                << setw(w) << "Metro-steps"
                << setw(w) << "Eq-steps"
                << setw(w) << "StepLength"
                << setw(w) << "Time"
                << setw(w) << "Analytical"
                << setw(w) << "Energy"
                << setw(w) << "Energy_std"
                << "\n";
    }
    else
    {
        outfile.open(filename, std::ios::out | std::ios::app);
    }
    outfile << setw(w) << m_numberOfDimensions
            << setw(w) << m_numberOfParticles
            << setw(w) << setprecision(5) << m_numberOfMetropolisSteps
            << setw(w) << setprecision(5) << numberOfEquilibrationSteps
            << setw(w) << fixed << setprecision(5) << m_stepLength
            << setw(w) << fixed << setprecision(0) << timing
            << setw(w) << analytical
            << setw(w) << setprecision(5) << m_energy
            << setw(w) << setprecision(5) << m_energy_std
            << "\n";

    outfile.close();
}
void Sampler::writeGradientSearchToFile(System &system, std::string filename, int epoch, std::vector<double> gradNorms, bool impoSamp, bool analytical, bool interaction, double learningRate)
{ // write out the gradient search to file
    // break filename to add "detailed_" to the beginning, after the path
    std::string path = filename.substr(0, filename.find_last_of("/\\") + 1);
    std::string filename_only = filename.substr(filename.find_last_of("/\\") + 1);
    filename = path + "detailed_" + filename_only + ".txt";
    std::ifstream exsists_file(filename.c_str());

    std::fstream outfile;
    // auto pa = system.getWaveFunctionParameters();
    // int p = pa.size();
    int w = 20;

    if (!exsists_file.good())
    {
        outfile.open(filename, std::ios::out);
        outfile << setw(w) << "Dimensions"
                << setw(w) << "Particles"
                << setw(w) << "Hidden-nodes"
                << setw(w) << "Metro-steps"
                << setw(w) << "StepLength";
        // for (int i = 0; i < p; i++)
        //     outfile << setw(w - 1) << "WF" << (i + 1);
        outfile << setw(w) << "Energy"
                << setw(w) << "Energy_std"
                << setw(w) << "Energy_var"
                << setw(w) << "Accept_number"
                << setw(w) << "Accept_ratio"
                << setw(w) << "Imposampling"
                << setw(w) << "Analytical"
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
            << setw(w) << setprecision(5) << m_numberOfMetropolisSteps
            << setw(w) << fixed << setprecision(5) << m_stepLength
            << setw(w) << fixed << setprecision(5) << m_energy
            << setw(w) << fixed << setprecision(5) << m_energy_std
            << setw(w) << fixed << setprecision(5) << m_energy_variance
            << setw(w) << fixed << setprecision(5) << m_numberOfAcceptedSteps
            << setw(w) << fixed << setprecision(5) << m_acceptRatio
            << setw(w) << fixed << setprecision(5) << impoSamp
            << setw(w) << fixed << setprecision(5) << analytical
            << setw(w) << fixed << setprecision(5) << interaction
            << setw(w) << fixed << setprecision(5) << epoch
            << setw(w) << fixed << setprecision(5) << learningRate
            << setw(w) << fixed << setprecision(5) << gradNorms.at(0)
            << setw(w) << fixed << setprecision(5) << gradNorms.at(1)
            << setw(w) << fixed << setprecision(5) << gradNorms.at(2)
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