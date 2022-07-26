import numpy as np
from scipy.constants import c, e, m_p, epsilon_0, m_e
from joblib import Parallel, delayed
import sys

import PyHEADTAIL
from PyHEADTAIL.trackers.detuners import Chromaticity, AmplitudeDetuning
import PyHEADTAIL.particles.generators as generators
from PyHEADTAIL.trackers.longitudinal_tracking import RFSystems
from PyHEADTAIL.impedances.wakes import WakeField, WakeTable, ResistiveWall, CircularResistiveWall
from PyHEADTAIL.particles import slicing
from tqdm import tqdm

from helper_funcs import *


def run(chromaticity):
    np.random.seed(42)
    n_turns = 2049
    n_turns_slicemonitor = 2048
    n_macroparticles = int(10**5)
    intensity = 5e10
    n_segments = 1
    Ekin = 11.4e6
    gamma = 1 + Ekin * e / (m_p * c**2)
    C = 216.72
    A = 238
    Z = 28
    Q_x = 4.2
    Q_y = 3.4
    gamma_t = 4.9
    alpha_0 = [gamma_t**-2]
    alpha_x_inj = 0.
    alpha_y_inj = 0.
    beta_x_inj = C/(2*np.pi*Q_x)
    beta_y_inj = C/(2*np.pi*Q_y)
    sigma_z = 4
    epsn_x = 12.5e-6  # [m rad]
    epsn_y = 12.5e-6  # [m rad]
    phi = 0 if (gamma**-2-gamma_t**-2) < 0 else np.pi
    h_rf = 4
    long_map = RFSystems(C, [h_rf, ], [16e3, ], [phi, ],
                         alpha_0, gamma, mass=A*m_p, charge=Z*e)
    print(long_map.Q_s)
    beta = np.sqrt(1 - gamma**-2)
    R = C / (2.*np.pi)

    p0 = np.sqrt(gamma**2 - 1) * A * m_p * c
    s = np.arange(0, n_segments + 1) * C / n_segments
    alpha_x = alpha_x_inj * np.ones(n_segments)
    beta_x = beta_x_inj * np.ones(n_segments)
    D_x = np.zeros(n_segments)
    alpha_y = alpha_y_inj * np.ones(n_segments)
    beta_y = beta_y_inj * np.ones(n_segments)
    D_y = np.zeros(n_segments)

    egeox = epsn_x / (beta * gamma)
    egeoy = epsn_y / (beta * gamma)
    bunches = [generators.ParticleGenerator(macroparticlenumber=n_macroparticles, intensity=intensity, charge=Z*e,
                                            gamma=gamma, mass=A*m_p, circumference=C,
                                            distribution_x=generators.gaussian2D(egeox), alpha_x=alpha_x, beta_x=beta_x,
                                            distribution_y=generators.gaussian2D(egeoy), alpha_y=alpha_y, beta_y=beta_y,
                                            limit_n_rms_x=3.5, limit_n_rms_y=3.5,
                                            distribution_z=generators.RF_bucket_distribution(
                                                long_map.get_bucket(gamma=gamma), sigma_z=sigma_z)
                                            ).generate() for i in range(h_rf)]
    for n, bunch in enumerate(bunches):
        bunch.z[:] += (n*C/h_rf)
        # print(bunch.sigma_z())
    # print(C/h_rf)
    bunch1, bunch2, bunch3, bunch4 = bunches
    # print(bunch1.mean_z(), bunch1.sigma_z())
    # print(bunch2.mean_z(), bunch2.sigma_z())
    # print(bunch3.mean_z(), bunch3.sigma_z())
    # print(bunch4.mean_z(), bunch4.sigma_z())
    allbunches = bunch1+bunch2+bunch3+bunch4
    # print(allbunches.mean_z(), allbunches.sigma_z())
    bunch_monitors = []
    slice_monitors = []
    for n in range(h_rf):
        folder = '/home/vgubaidulin/PhD/Data/CoupledBunch/bunch{0:}'.format(
            n+1)
        bunch_monitor = get_bunch_monitor(folder, chromaticity, n_turns)
        n_slices = 50
        slicer = slicing.UniformBinSlicer(n_slices=n_slices, n_sigma_z=4)
        slice_monitor = get_slice_monitor(
            folder, chromaticity, n_turns_slicemonitor, slicer)
        bunch_monitors.append(bunch_monitor)
        slice_monitors.append(slice_monitor)
    n_wake_slices = 500
    dt_min = C/c/n_wake_slices
    res_wall1 = CircularResistiveWall(pipe_radius=68e-3,
                                      resistive_wall_length=3*30.72,
                                      dt_min=dt_min,
                                      conductivity=1.4e6)
    res_wall2 = CircularResistiveWall(pipe_radius=49e-3,
                                      resistive_wall_length=3*62.88,
                                      dt_min=dt_min,
                                      conductivity=1.4e6)
    res_wall3 = CircularResistiveWall(pipe_radius=0.1,
                                      resistive_wall_length=3*123.12,
                                      dt_min=dt_min,
                                      conductivity=1.4e6)
    z_cuts = (-4.2*bunch1.sigma_z(), 3*C/h_rf+4.2*bunch4.sigma_z())
    print(z_cuts)
    wake_slicer = slicing.UniformBinSlicer(
        n_slices=n_wake_slices, z_cuts=z_cuts)
    wake_field = WakeField(wake_slicer, res_wall1, res_wall2, res_wall3)
    folderall = '/home/vgubaidulin/PhD/Data/CoupledBunch/allbunches'
    allslices_monitor = get_slice_monitor(
        folderall, chromaticity, n_turns_slicemonitor, wake_slicer)
    chroma = Chromaticity(Qp_x=[chromaticity*Q_x], Qp_y=[chromaticity*Q_y])
    trans_map = TransverseMap(s, alpha_x, beta_x, D_x,
                              alpha_y, beta_y, D_y, Q_x, Q_y, [chroma])
    trans_one_turn = [m for m in trans_map]
    map_ = trans_one_turn + [long_map]
    for turn in tqdm(range(n_turns)):
        for m_ in map_:
            m_.track(allbunches)
            # m_.track(bunch2)
            # m_.track(bunch3)
            # m_.track(bunch4)
            # wake_field.track(allbunches)
        # bunch1, bunch2, bunch3, bunch4 = allbunches
        # for i, bunch_monitor in enumerate(bunch_monitors):
            # bunch_monitor.dump(bunches[i])
            # if (turn >= n_turns - n_turns_slicemonitor):
            # slice_monitors[i].dump(bunches[i])
        allslices_monitor.dump(allbunches)


if __name__ == '__main__':
    run(0)
