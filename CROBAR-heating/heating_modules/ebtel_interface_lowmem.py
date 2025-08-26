import copy
import numpy as np
from synthesizAR.interfaces import ebtel
import os

class ebtel_interface_lowmem(ebtel.EbtelInterface):
	def configure_input(self, loop):

		event_properties = self.heating_model.calculate_event_properties(loop)

		output_filename = os.path.join(self.config_dir, loop.name+'.xml')
		output_dict = copy.deepcopy(self.base_config)
		output_dict['output_filename'] = os.path.join(self.results_dir, loop.name)
		output_dict['loop_length'] = event_properties['loop_length'] # loop.full_length.to(u.cm).value / 2.0
		output_dict['config_filename'] = output_filename
		output_key_names = ['total_time','tau','tau_max','loop_length','saturation_limit','force_single_fluid','use_c1_loss_correction',
				'use_c1_grav_correction','use_power_law_radiative_losses','use_flux_limiting','calculate_dem','save_terms',
				'use_adaptive_solver','output_filename','adaptive_solver_error','adaptive_solver_safety','c1_cond0','c1_rad0',
				'helium_to_hydrogen_ratio','surface_gravity']

		decay_ends = event_properties['decay_end'].value.astype(np.dtype('U25'))
		decay_starts = event_properties['decay_start'].value.astype(np.dtype('U25'))
		magnitudes = event_properties['magnitude'].value.astype(np.dtype('U25'))
		rise_ends = event_properties['rise_end'].value.astype(np.dtype('U25'))
		rise_starts = event_properties['rise_start'].value.astype(np.dtype('U25'))

		f = open(output_filename,'w')
		f.write('<?xml version="1.0" ?>\n<root>\n')
		for i in range(len(output_key_names)):
			f.write('    <'+output_key_names[i]+'>'+str(output_dict[output_key_names[i]])+'</'+output_key_names[i]+'>\n')
		f.write('    <dem>\n')
		f.write('        <use_new_method>'+str(output_dict['dem']['use_new_method'])+'</use_new_method>\n')
		f.write('        <temperature bins="'+str(output_dict['dem']['temperature']['bins'])
						+'" log_max="'+str(output_dict['dem']['temperature']['log_max'])
						+'" log_min="'+str(output_dict['dem']['temperature']['log_min'])+'"/>\n')
		f.write('    </dem>\n')
		f.write('    <heating>\n')
		f.write('        <background>'+str(output_dict['heating']['background'])+'</background>\n')
		f.write('        <partition>'+str(output_dict['heating']['partition'])+'</partition>\n')
		f.write('        <events>\n')
		for i in range(event_properties['magnitude'].shape[0]):
			f.write('            <event decay_end="'+decay_ends[i]+' s" decay_start="'
					+decay_starts[i]+' s" magnitude="'+magnitudes[i]+' erg / (cm3 s)" rise_end="'
					+rise_ends[i]+' s" rise_start="'+rise_starts[i]+' s"/>\n')
		f.write('        </events>\n')
		f.write('    </heating>\n')
		f.write('    <config_filename>'+output_dict['config_filename']+'</config_filename>\n')
		f.write('</root>')
		f.close

		hydro_config = copy.deepcopy(output_dict)
		hydro_config['events'] = event_properties
		loop.hydro_configuration = hydro_config

	#@processify
	def load_results(self, loop):
		"""
		Load EBTEL output for a given loop object.

		Parameters
		----------
		loop : `synthesizAR.Loop` object
		"""
		# load text
		N_s = loop.field_aligned_coordinate.shape[0]
		_tmp = np.loadtxt(loop.hydro_configuration['output_filename'])

		# reshape into a 1D loop structure with units
		time = _tmp[:, 0]
		electron_temperature = _tmp[:, 1].astype('float32')
		ion_temperature = _tmp[:, 2].astype('float32')
		density = _tmp[:, 3].astype('float32')
		velocity = _tmp[:, -2].astype('float32')
		# flip sign of velocity where the radial distance from center is maximum
		# FIXME: this is probably not the best way to do this...
		#r = np.sqrt(np.sum(loop.coordinate.cartesian.xyz.value**2, axis=0))
		#i_mirror = np.where(np.diff(np.sign(np.gradient(r))))[0]
		#if i_mirror.shape[0] > 0:
		#	i_mirror = i_mirror[0] + 1
		#else:
		#	# If the first method fails, just set it at the midpoint
		#	i_mirror = int(N_s / 2) if N_s % 2 == 0 else int((N_s - 1) / 2)
		#velocity[:, i_mirror:] = -velocity[:, i_mirror:]

		return time, electron_temperature, ion_temperature, density, velocity
