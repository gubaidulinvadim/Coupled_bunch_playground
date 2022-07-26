import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbs
from scipy.constants import c, e, m_p, epsilon_0, m_e
from joblib import Parallel, delayed
from tqdm import tqdm
import sys

import PyHEADTAIL
from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.longitudinal_tracking import LinearMap
from PyHEADTAIL.trackers.detuners import Chromaticity, AmplitudeDetuning
from PyHEADTAIL.elens.elens import ElectronLens
from PyHEADTAIL.spacecharge.spacecharge import TransverseGaussianSpaceCharge
from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor, ParticleMonitor
from PyHEADTAIL.impedances import wakes, wake_kicks
from PyHEADTAIL.aperture.aperture import CircularApertureXY
from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
import PyHEADTAIL.particles.generators as generators
from PyHEADTAIL.particles import slicing
from PyHEADTAIL.multipoles.multipoles import ThinOctupole
from scipy.fftpack import fft, fftfreq
import PyNAFF as pnf
def get_bunch_monitor(folder, chroma, n_turns, parameters_dict=None):
    filename = folder+'BM(chroma={0:.3f})'.format(chroma)
    bunch_monitor = BunchMonitor(
        filename=filename, n_steps=n_turns, 
        parameters_dict=parameters_dict
        )
    return bunch_monitor
def get_slice_monitor(folder, chroma, n_turns, slicer, parameters_dict=None):
    filename = folder+'SLM(chroma={0:.3f})'.format(chroma)
    slice_monitor = SliceMonitor(filename=filename, n_steps=n_turns, slicer=slicer, parameters_dict=parameters_dict)
    return slice_monitor
def get_wake_monitor(folder, chroma, n_turns, slicer, parameters_dict=None):
    filename = folder+'WM(chroma={0:.3f})'.format(chroma)
    slice_monitor = SliceMonitor(filename=filename, n_steps=n_turns, slicer=slicer, parameters_dict=parameters_dict)
    return slice_monitor
def get_particle_monitor(folder, chroma, n_turns, parameters_dict=None):
    filename = folder+'PM(chroma={0:.3f})'.format(chroma)
    particle_monitor = ParticleMonitor(filename=filename, stride=64, n_steps=n_turns, parameters_dict=parameters_dict)
    return particle_monitor
