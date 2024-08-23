# containers for model/beamline data
# for internal use by BmadLiveModel to publish model data as object attributes


import os
import sys
import numpy as np
from dataclasses import dataclass
from scipy import constants


Q_ELECTRON_C = constants.elementary_charge
MASS_ELECTRON_MEV = constants.physical_constants['electron mass energy equivalent in MeV'][0]
MASS_ELECTRON_EV = MASS_ELECTRON_MEV * 1e6

PATH_SELF = os.path.dirname(os.path.abspath(__file__))
DIR_SELF = os.path.join(*os.path.split(PATH_SELF)[:-1])
sys.path.append(DIR_SELF)


# support dataclass for twiss data, so user can type things like 'twiss.beta_x'
@dataclass
class _Twiss:
    beta_x: np.array
    beta_y: np.array
    alpha_x: np.array
    alpha_y: np.array
    eta_x: np.array
    eta_y: np.array
    etap_x: np.array
    etap_y: np.array
    psi_x: np.array
    psi_y: np.array

    @property
    def gamma_x(self): return _gamma_twiss(self.alpha_x, self.beta_x)

    @property
    def gamma_y(self): return _gamma_twiss(self.alpha_y, self.beta_y)
    

def _gamma_twiss(alpha, beta): return (1 + alpha**2) / beta


# containers for model elements, nonexhaustive

@dataclass
class _Quad:
    S: float
    l: float
    b1_gradient: float

    @property
    def b1(self): return self.b1_gradient * self.l

@dataclass
class _Corrector:
    S: float
    l: float
    kick: float = 0.0

@dataclass
class _Dipole:
    S: float
    l: float
    b_field: float

@dataclass
class _Cavity:
    S: float
    l: float
    voltage: float
    phase: float

    @property
    def gradient(self): return self.voltage / self.l


# beamline data
class _ModelData:
    def __init__(self, p0c, e_tot, twiss):
        self.p0c, self.e_tot, self.twiss = p0c, e_tot, twiss
        self.rf, self.quads, self.bends = {}, {}, {}
        # self.xcors, self.ycors = {}, {}
        self.cors = {}

    @property
    def gamma_rel(self): return self.p0c / MASS_ELECTRON_EV

    @property
    def Brho(self): return self.p0c / Q_ELECTRON_C