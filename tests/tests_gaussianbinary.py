# import directory
import sys
import subprocess

sys.path.append('../Analysis/')
import pytest

import cpp_utils
import numpy as np



class CppFunctions:
    # constructor of the class
    def __init__(
        self,
        NumberParticles,
        NumberDimensions,
        NumberHidden,
        TimeStep,
        equilibration,
        Stepper,
        Importance,
        interaction,
    ):
        self.NumberParticles = NumberParticles
        self.NumberDimensions = NumberDimensions
        self.NumberHidden = NumberHidden
        self.TimeStep = TimeStep
        self.equilibration = equilibration
        self.Stepper = Stepper
        self.Importance = Importance
        self.interaction = interaction


    def runCpp(self):
        tests_path = cpp_utils.testsPath()

        assert (
            tests_path.exists()
        ), f'I cannot find {tests_path} :((, are you sure you have compiled?'
        args = [
            tests_path,
            self.NumberDimensions,
            self.NumberParticles,
            self.NumberHidden,
            int(self.interaction),
        ]

        args_run = [str(arg) for arg in args]
        subprocess.run(args_run)    


    def Qfac(self):
        ## Read in data
        Q = np.loadtxt("Data/testQfac.txt")
        return Q

    def evaluate(self):
        ## Read in data
        wf_eval = np.loadtxt("Data/wf_eval.txt")
        return wf_eval

    def computeHidBiasDerivative(self):
        ## Read in data
        hid_bias_der = np.loadtxt("Data/hidBiasDer.txt")
        return hid_bias_der

    def computeVisBiasDerivative(self):
        ## Read in data
        vis_bias_der = np.loadtxt("Data/visBiasDer.txt")
        return vis_bias_der

    def computeWeightsDerivative(self):
        ## Read in data
        weights_der = np.loadtxt("Data/weightsDer.txt")
        return weights_der

    def computeLocalEnergy(self):
        ## Read in data
        local_energy = np.loadtxt("Data/localEnergy.txt")
        return local_energy

class PythonFunctions:

    # constructor of the class
    def __init__(
        self,
        NumberParticles,
        NumberDimensions,
        NumberHidden,
        TimeStep,
        equilibration,
        Stepper,
        Importance,
        interaction,
    ):
        self.NumberParticles = NumberParticles
        self.NumberDimensions = NumberDimensions
        self.NumberHidden = NumberHidden
        self.TimeStep = TimeStep
        self.equilibration = equilibration
        self.Stepper = Stepper
        self.Importance = Importance
        self.interaction = interaction
    

    def WaveFunction(self, r, a, b, w):
        sigma = 1.0
        sig2 = sigma**2
        Psi1 = 0.0
        Psi2 = 1.0
        Q = self.Qfac(r, b, w)

        for iq in range(self.NumberParticles):
            for ix in range(self.NumberDimensions):
                Psi1 += (r[iq, ix] - a[iq, ix]) ** 2

        for ih in range(self.NumberHidden):
            Psi2 *= 1.0 + np.exp(Q[ih])

        Psi1 = np.exp(-Psi1 / (2 * sig2))

        return Psi1 * Psi2

    # Local energy  for the 2-electron quantum dot in two dims, using analytical local energy
    def LocalEnergy(self, r, a, b, w):
        sigma = 1.0
        sig2 = sigma**2
        locenergy = 0.0

        Q = self.Qfac(r, b, w)

        for iq in range(self.NumberParticles):
            for ix in range(self.NumberDimensions):
                sum1 = 0.0
                sum2 = 0.0
                for ih in range(self.NumberHidden):
                    sum1 += w[iq, ix, ih] / (1 + np.exp(-Q[ih]))
                    sum2 += (
                        w[iq, ix, ih] ** 2
                        * np.exp(Q[ih])
                        / (1.0 + np.exp(Q[ih])) ** 2
                    )
                dlnpsi1 = -(r[iq, ix] - a[iq, ix]) / sig2 + sum1 / sig2
                dlnpsi2 = -1 / sig2 + sum2 / sig2**2
                locenergy += 0.5 * (
                    -dlnpsi1 * dlnpsi1 - dlnpsi2 + r[iq, ix] ** 2
                )

        if self.interaction == True:
            print("Interaction is on")
            for iq1 in range(self.NumberParticles):
                for iq2 in range(iq1):
                    distance = 0.0
                    for ix in range(self.NumberDimensions):
                        distance += (r[iq1, ix] - r[iq2, ix]) ** 2

                    locenergy += 1 / np.sqrt(distance)

        return locenergy

    # Derivate of wave function ansatz as function of variational parameters
    def DerivativeWFansatz(self, r, a, b, w):

        sigma = 1.0
        sig2 = sigma**2

        Q = self.Qfac(r, b, w)

        WfDer = np.empty((3,), dtype=object)
        WfDer = [np.copy(a), np.copy(b), np.copy(w)]

        WfDer[0] = (r - a) / sig2
        WfDer[1] = 1 / (1 + np.exp(-Q))

        for ih in range(self.NumberHidden):
            WfDer[2][:, :, ih] = w[:, :, ih] / (sig2 * (1 + np.exp(-Q[ih])))

        return WfDer

    # Setting up the quantum force for the two-electron quantum dot, recall that it is a vector
    def QuantumForce(self, r, a, b, w):

        sigma = 1.0
        sig2 = sigma**2

        qforce = np.zeros((self.NumberParticles, self.NumberDimensions), np.double)
        sum1 = np.zeros((self.NumberParticles, self.NumberDimensions), np.double)

        Q = self.Qfac(r, b, w)

        for ih in range(self.NumberHidden):
            sum1 += w[:, :, ih] / (1 + np.exp(-Q[ih]))

        qforce = 2 * (-(r - a) / sig2 + sum1 / sig2)

        return qforce

    def Qfac(self, r, b, w):
        Q = np.zeros((self.NumberHidden), np.double)
        temp = np.zeros((self.NumberHidden), np.double)

        for ih in range(self.NumberHidden):
            temp[ih] = (r * w[:, :, ih]).sum()

        Q = b + temp

        return Q


def test_Qfac(r, b, w, pyfunc, cppfunc):

    pyQfac = pyfunc.Qfac(r, b, w)
    cppQ = cppfunc.Qfac()

    assert(np.abs((pyQfac - cppQ)).sum() < 1e-10) 

    #assert pyfunc.Qfac(r, b, w) == cppfunc.Qfac(r, b, w)

def test_evaluate(r, b, w, pyfunc, cppfunc):
    pyWF = pyfunc.WaveFunction(r, a, b, w)
    cppWF = cppfunc.evaluate()

    assert(np.abs((pyWF - cppWF)).sum() < 1e-10)

def test_computeVisBiasDerivative(r, b, w, pyfunc, cppfunc):
    pyDer = pyfunc.DerivativeWFansatz(r, a, b, w)[0]
    cppDer = cppfunc.computeVisBiasDerivative()
    #print(pyDer)
    #print(cppDer)

    assert(np.abs((pyDer - cppDer)).sum() < 1e-10)

def test_computeHidBiasDerivative(r, b, w, pyfunc, cppfunc):
    pyDer = pyfunc.DerivativeWFansatz(r, a, b, w)[1]
    cppDer = cppfunc.computeHidBiasDerivative()
    #print(pyDer)
    #print(cppDer)

    assert(np.abs((pyDer - cppDer)).sum() < 1e-10)

def test_computeWeightsDerivative(r, b, w, pyfunc, cppfunc):
    pyDer = pyfunc.DerivativeWFansatz(r, a, b, w)[2]
    cppDer = cppfunc.computeWeightsDerivative().reshape(NumberParticles, NumberDimensions, NumberHidden)

    assert(np.abs((pyDer - cppDer)).sum() < 1e-10)

def test_localEnergy(r, a, b, w, pyfunc, cppfunc):
    pyE = pyfunc.LocalEnergy(r, a, b, w)
    cppE = cppfunc.computeLocalEnergy()

    print(pyE)
    print(cppE)

    assert(np.abs((pyE - cppE)) < 1e-10)



# def testquantumForce:


if __name__ == '__main__':
 # Global variables
    NumberParticles = 2
    NumberDimensions = 2
    NumberHidden = 2
    TimeStep = 0.01
    equilibration = 0.1
    Stepper = 0
    Importance = 0
    interaction = False

    cppfunc = CppFunctions(
        NumberParticles,
        NumberDimensions,
        NumberHidden,
        TimeStep,
        equilibration,
        Stepper,
        Importance,
        interaction,
    )

    cppfunc.runCpp()

    a = np.loadtxt("Data/a.txt").reshape(NumberParticles, NumberDimensions)
    b = np.loadtxt("Data/b.txt").reshape(NumberHidden)
    w = np.loadtxt("Data/w.txt").reshape(NumberParticles, NumberDimensions, NumberHidden)
    r = np.loadtxt("Data/wfpos.txt").reshape(NumberParticles, NumberDimensions)
    
    pyfunc = PythonFunctions(
        NumberParticles,
        NumberDimensions,
        NumberHidden,
        TimeStep,
        equilibration,
        Stepper,
        Importance,
        interaction,
    )

    cppfunc = CppFunctions(
        NumberParticles,
        NumberDimensions,
        NumberHidden,
        TimeStep,
        equilibration,
        Stepper,
        Importance,
        interaction,
    )

    test_Qfac(r, b, w, pyfunc, cppfunc)
    test_evaluate(r, b, w, pyfunc, cppfunc)
    test_computeVisBiasDerivative(r, b, w, pyfunc, cppfunc)
    test_computeHidBiasDerivative(r, b, w, pyfunc, cppfunc)
    test_computeWeightsDerivative(r, b, w, pyfunc, cppfunc)
    test_localEnergy(r, a, b, w, pyfunc, cppfunc)

    # testquantumForce()
    pass
