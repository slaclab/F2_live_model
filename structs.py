# containers for model/beamline data
# for internal use by BmadLiveModel to publish model data as object attributes


import os
import sys
import numpy as np
from dataclasses import dataclass
from scipy import constants

C = MASS_ELECTRON_MEV = constants.speed_of_light
Q_ELECTRON_C = constants.elementary_charge
MASS_ELECTRON_MEV = constants.physical_constants['electron mass energy equivalent in MeV'][0]
MASS_ELECTRON_EV = MASS_ELECTRON_MEV * 1e6

PATH_SELF = os.path.dirname(os.path.abspath(__file__))
DIR_SELF = os.path.join(*os.path.split(PATH_SELF)[:-1])
sys.path.append(DIR_SELF)


# support dataclass for twiss data, so user can type things like 'twiss.beta_x'
@dataclass
class _Twiss:
    beta_x: np.ndarray
    beta_y: np.ndarray
    alpha_x: np.ndarray
    alpha_y: np.ndarray
    eta_x: np.ndarray
    eta_y: np.ndarray
    etap_x: np.ndarray
    etap_y: np.ndarray
    psi_x: np.ndarray
    psi_y: np.ndarray

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

# containers for LEM data
# intended interface goes like f2m.LEM.L0.amplitude or f2m.LEM.L2.BDES[:]

@dataclass
class _LEMRegionData():

    def __init__(self, N_elems, amplitude=1.0, chirp=0.0, fudge=1.0):
        """
        """
        self.amplitude = amplitude
        self.chirp = chirp
        self.fudge = fudge
        self.elements = np.ndarray(N_elems, dtype='U20')
        self.device_names = np.ndarray(N_elems, dtype='U20')
        self.S = np.ndarray(N_elems)
        self.Z = np.ndarray(N_elems)
        self.L = np.ndarray(N_elems)
        self.BDES = np.ndarray(N_elems)
        self.BLEM = np.ndarray(N_elems)
        self.BLEM_save = np.ndarray(N_elems)
        self.EREF = np.ndarray(N_elems)
        self.EACT = np.ndarray(N_elems)
        self.EERR = np.ndarray(N_elems)

@dataclass
class _F2LEMData:
    L0: _LEMRegionData
    L1: _LEMRegionData
    L2: _LEMRegionData
    L3: _LEMRegionData

class _ModelData:
    def __init__(self, p0c, e_tot, twiss):
        """
        (private) data container for non-static model parameters

        data that are defined for every element are provided as np.arrays:
        ``p0c, e_tot, twiss.<attr> `` 

        per-device data are provided as name-indexed dictionaries
        """
        self.p0c = p0c     #: z-momentum profile in eV
        self.e_tot = e_tot #: total particle energy in eV
        self.twiss = twiss #: _Twiss dataclass
        self.rf = {}       #: dictionary of _Cavity objects, indexed by element name
        self.quads = {}    #: dictionary of _Quad objects, indexed by element name
        self.bends = {}    #: dictionary of _Dipole objects, indexed by element name

    @property
    def gamma_rel(self):
        """ relativistic Lorentz factor """
        return self.p0c / MASS_ELECTRON_EV

    @property
    def Brho(self):
        """
        particle magnetic rigidity (for electrons) in kG m :
        Brho ~ 10 * E [MeV] / 300 = 10 * E [eV] / 0.3e9 (c)
        """
        return 10 * self.p0c / 0.299792e9
