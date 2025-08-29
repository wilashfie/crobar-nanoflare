import os
from sys import path
import argparse
import numpy as np

from pydrad.configure import Configure
from pydrad.configure.util import get_clean_hydrad, run_shell_command
from pydrad.configure.data import get_defaults
from astropy import units as u

from util import parse_xml_config, get_events




def run_hydrad_simulation(tmpdir):

    base_path = os.getcwd()
    path.append(os.path.join(base_path, 'modules'))
    
    # clean HYDRAD installation
    hydrad_clean = os.path.join(tmpdir, 'hydrad-clean')
    if not os.path.exists(hydrad_clean):
        get_clean_hydrad(hydrad_clean, from_github=True)

    ebtel_loopfile = 'ebtel_runs/heat_config_runs_h140_g0.0/hydro_config/loop000456.xml'

    config_data = parse_xml_config(ebtel_loopfile)

    loop_length = (config_data['loop_length']*u.cm).to(u.Mm)

    location = 0.5
    scale_height = 0.25

    events = get_events(config_data['heating'],loop_length,location,scale_height)

    ttot = config_data['total_time']
    #ttot= 3600*2 #  3600

   
    #bgrd = {'use_initial_conditions':False, 'location':0.5*ll*u.cm, 'scale_height': 1e300*u.cm, 'rate':bhrate*u.erg/u.s/u.cm**3}
    bgrd = {'use_initial_conditions':True}

    # configuration
    config = get_defaults()
    config['general']['total_time'] = ttot * u.s
    config['general']['output_interval'] = 25 * u.s
    config['general']['loop_length'] = loop_length
    config['general']['footpoint_height'] = 3 * u.Mm

    config['heating']['background'] = bgrd

    config['heating']['events'] = events
    #config['heating']['events'] = [heat1,heat2]

    config['initial_conditions']['footpoint_temperature'] = 20000*u.K
    config['initial_conditions']['footpoint_density'] = 1.0e10*u.cm**(-3)

    #config['grid']['initial_refinement_level'] = 6
    #config['grid']['maximum_refinement_level'] = 6
    #config['grid']['maximum_cell_width'] = 0.2 * u.Mm
    #config['grid']['minimum_cells'] = 300
    config['radiation']['use_power_law_radiative_losses'] = True
    
    # configure and run simulation
    c = Configure(config)
    hydrad_results = os.path.join(tmpdir, 'steady-run-ebtel')
    c.setup_simulation(hydrad_results, hydrad_clean, overwrite=True)
    
    #run_shell_command(os.path.join(hydrad_results, 'HYDRAD.exe'))
    
    return hydrad_results


def main():

    parser = argparse.ArgumentParser(description="Run HYDRAD simulation with specified configuration.")
    parser.add_argument('--tmpdir', type=str, default='hydrad_test', help='Temporary directory for HYDRAD simulation')
    #parser.add_argument('--config_filename', type=str, default='config_test.json', help='Configuration filename for the simulation')
    args = parser.parse_args()

    #tmpdir = 'hydrad_test'
    #config_filename = 'config_test.json'
    
    try:
        results_path = run_hydrad_simulation(args.tmpdir)
        print(f"Simulation completed successfully. Results saved to: {results_path}")
    except Exception as e:
        print(f"Error running simulation: {e}")


if __name__ == "__main__":
    main()