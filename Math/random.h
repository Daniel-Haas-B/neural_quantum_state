#pragma once

#include <random>
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <cmath>
#include "./pcg-cpp-0.98/include/pcg_random.hpp"

/* This is a proposal for a convenience random number generator. However, feel
 * free to use the standard library, or any other libabry, to create your own
 * generators. Example usage of this class is:
 *
 *  auto rng = std::make_unique<Random>(2020); // Create a unique rng instance
 *  int foo = rng->nextInt(5, 10); // Draw random uniform integers in [5, 10]
 *  double bar = rng->nextDouble(); // Draw random uniform doubles in [0, 1)
 */

class Random
{

private:
    std::mt19937_64 m_mtEngine;
    pcg32 m_pcgEngine;

    // create an object that can be one engine or the other
    std::variant<std::mt19937_64, pcg32> m_engine;

public:
    Random(std::string rngType)
    {
        if (rngType == "mt")
        {
            std::random_device rd; // Will be used to obtain a seed for the random number engine
            m_mtEngine = std::mt19937_64(rd());
            m_engine = m_mtEngine;
        }
        else if (rngType == "pcg")
        {
            pcg_extras::seed_seq_from<std::random_device> seed_source; // this is a seed sequence
            pcg32 rng(seed_source);                                    // pcg32_random_t rng;
            m_pcgEngine = rng;
            m_engine = m_pcgEngine;
        }
        else
        {
            std::cout << "Please specify a valid random number generator type (mt or pcg)" << std::endl;
        }
    }

    Random(int seed, std::string rngType)
    {
        if (rngType == "mt")
        {
            m_mtEngine = std::mt19937_64(seed);
            m_engine = m_mtEngine;
        }
        else if (rngType == "pcg")
        {
            pcg32 rng(seed); // pcg32_random
            m_pcgEngine = rng;
            m_engine = m_pcgEngine;
        }
        else
        {
            std::cout << "Please specify a valid random number generator type (mt or pcg)" << std::endl;
        }
    }

    int nextInt(const int &lowerLimit, const int &upperLimit)
    {
        // Produces uniformly distributed random integers in the closed interval
        // [lowerLimit, upperLimit].

        std::uniform_int_distribution<int> dist(lowerLimit, upperLimit);

        return dist(m_mtEngine);
    }

    int nextInt(const int &upperLimit)
    {
        // Produces uniformly distributed random integers in the closed interval
        // [0, upperLimit].

        std::uniform_int_distribution<int> dist(0, upperLimit);
        // return dist(m_engine);
        return dist(m_mtEngine);
    }

    double nextDouble()
    {
        // Produces uniformly distributed random floating-point values in the
        // half-open interval [0, 1).

        std::uniform_real_distribution<double> dist(0, 1);
        //  return dist(m_engine);

        return dist(m_mtEngine);
    }

    double nextDouble(const int &lowerLimit, const int &upperLimit)
    {
        // Produces uniformly distributed random floating-point values in the
        // half-open interval [0, 1).

        std::uniform_real_distribution<double> dist(lowerLimit, upperLimit);
        //  return dist(m_engine);

        return dist(m_mtEngine);
    }
    double nextGaussian(
        const double &mean,
        const double &standardDeviation)
    {
        // Produces normal distributed random floating-point values with mean
        // ``mean`` and standard deviation ``standardDeviation``.

        std::normal_distribution<double> dist(mean, standardDeviation);
        // return dist(m_engine);

        return dist(m_mtEngine);
    }
};
