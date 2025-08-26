def load_terms(loop):
	import copy
	import numpy as np
	"""
	Load additional EBTEL terms given a loop object.

	Parameters
	----------
	loop : `synthesizAR.Loop` object
	"""
	# load text
	_tmp = np.loadtxt(loop.hydro_configuration['output_filename']+'.terms')

	# Terms file has following components:
	# Column 0: coronal electron radiative loss rate
	# Column 1: coronal ion radiative loss rate
	# Column 2: Ratio of coronal to transition radiative loss (EBTEL c1)
	# Column 3: Effective radiative loss coefficient
	# Radiative loss rates are per unit area.
	return copy.deepcopy({'rlecor':_tmp[:,0],'rlicor':_tmp[:,1],'c1':_tmp[:,2],'coef':_tmp[:,3],'name':loop.name})
