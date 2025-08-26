"""
Compare results of adaptive and static solvers
"""
from collections import OrderedDict

import pytest
import numpy as np

from .helpers import run_ebtelplusplus


@pytest.fixture
def base_config():
    base_config = {
        'total_time': 5e3,
        'tau': 1.0,
        'tau_max': 10.0,
        'loop_length': 4e9,
        'saturation_limit': 1/6,
        'force_single_fluid': False,
        'use_c1_loss_correction': True,
        'use_c1_grav_correction': True,
        'use_flux_limiting': True,
        'calculate_dem': True,
        'save_terms': False,
        'adaptive_solver_error': 1e-6,
        'adaptive_solver_safety': 0.5,
        'c1_cond0': 2.0,
        'c1_rad0': 0.6,
        'helium_to_hydrogen_ratio': 0.075,
        'surface_gravity': 1.0,
        'heating': OrderedDict({
            'partition': 1.0,
            'background': 1e-6,
            'events': [
                {'event': {'rise_start': 0.0, 'rise_end': 100.0, 'decay_start': 100.0,
                           'decay_end': 200.0, 'magnitude': 0.1}}
            ]
        }),
        'dem': OrderedDict({
            'use_new_method': True,
            'temperature': {'bins': 451, 'log_min': 4, 'log_max': 8.5},
        }),
    }
    return base_config


@pytest.fixture
def adaptive_results(base_config, tmpdir):
    config = base_config.copy()
    config['use_adaptive_solver'] = True
    return run_ebtelplusplus(config)    


@pytest.fixture
def static_results(base_config, tmpdir):
    config = base_config.copy()
    config['use_adaptive_solver'] = False
    return run_ebtelplusplus(config)


def test_temperature_equal_adaptive_static(adaptive_results, static_results):
    t_static = static_results['time']
    t_adaptive = adaptive_results['time']
    adapt_interp = np.interp(t_static, t_adaptive, adaptive_results['electron_temperature'])
    # NOTE: Skip the first 5 steps b/c there is always one anomalous point that gives
    # an error > 10%; due to static case not rising fast enough
    assert np.allclose(adapt_interp[5:],
                       static_results['electron_temperature'][5:], rtol=1e-2, atol=1e4)
    adapt_interp = np.interp(t_static, t_adaptive, adaptive_results['ion_temperature'])
    assert np.allclose(adapt_interp[5:], static_results['ion_temperature'][5:], rtol=1e-2, atol=1e4)


def test_density_equal_adaptive_static(adaptive_results, static_results):
    t_static = static_results['time']
    t_adaptive = adaptive_results['time']
    adapt_interp = np.interp(t_static, t_adaptive, adaptive_results['density'])
    assert np.allclose(adapt_interp[5:], static_results['density'][5:], rtol=1e-2, atol=1e7)


def test_pressure_equal_adaptive_static(adaptive_results, static_results):
    t_static = static_results['time']
    t_adaptive = adaptive_results['time']
    adapt_interp = np.interp(t_static, t_adaptive, adaptive_results['electron_pressure'])
    assert np.allclose(adapt_interp[5:], static_results['electron_pressure'][5:],
                       rtol=1e-2, atol=1e-4)
    adapt_interp = np.interp(t_static, t_adaptive, adaptive_results['ion_pressure'])
    assert np.allclose(adapt_interp[5:], static_results['ion_pressure'][5:], rtol=1e-2, atol=1e-4)


def test_velocity_equal_adaptive_static(adaptive_results, static_results):
    t_static = static_results['time']
    t_adaptive = adaptive_results['time']
    adapt_interp = np.interp(t_static, t_adaptive, adaptive_results['velocity'])
    assert np.allclose(adapt_interp[5:], static_results['velocity'][5:], rtol=1e-2, atol=1e4)
