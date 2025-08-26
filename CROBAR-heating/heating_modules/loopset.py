"""
Holds a set of 3D field lines for interfacing with 1D Hydrodynamic simulations
Similar to SynthesizAR skeleton, but more focused on the container and interface
role, without all the added visualization and similar features
"""

import numpy as np
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator

def field_interp_setup(Bfield):
	Bdomain = np.vstack(np.array((Bfield.domain_left_edge,Bfield.domain_right_edge)))
	Bd_size = Bdomain[1,:]-Bdomain[0,:]
	Bd_ndim = Bfield.domain_dimensions
	Bd_dvox = Bd_size/(Bd_ndim-1.0)
	xa = Bdomain[0,0]+np.arange(Bd_ndim[0])*Bd_dvox[0]
	ya = Bdomain[0,1]+np.arange(Bd_ndim[1])*Bd_dvox[1]
	za = Bdomain[0,2]+np.arange(Bd_ndim[2])*Bd_dvox[2]
	Bx = np.array(Bfield.r['Bx'].reshape(Bd_ndim))
	By = np.array(Bfield.r['By'].reshape(Bd_ndim))
	Bz = np.array(Bfield.r['Bz'].reshape(Bd_ndim))
	B2 = np.array(Bx*Bx+By*By+Bz*Bz).astype('float32')
	return RegularGridInterpolator((xa,ya,za),B2**0.5,fill_value=None,bounds_error=False)
	
def field_interp(coords,interpolator):
	xa=coords[:,0]
	ya=coords[:,1]
	za=coords[:,2]
	return interpolator((xa,ya,za))
	
class loopset(object):	

	def __init__(self,loops):
		self.loops=loops
		
	@classmethod
	def get_loops(cls, lines, mags, lengths, loopnames=None):
		nloops = len(mags)
		#interpolator = field_interp_setup(Bfield)
		if(loopnames == None): loopnames = [None]*nloops
		loops=[]
		for i in range (0,nloops): 
			loops.append(loopobj(i,lines,mags,lengths,loopnames[i]))
		return cls(loops)

	def configure_loop_simulations(self, interface, **kwargs):
		"""
		Configure hydrodynamic simulations for each loop object
		"""
		for loop in self.loops:
			interface.configure_input(loop, **kwargs)

	def load_loop_simulations(self,interface,filename):
		for loop in self.loops:
			(time,electron_temperature,ion_temperature,density,velocity) = interface.load_results(loop)
			loop.time=time
			loop.electron_temperature=electron_temperature
			loop.ion_temperature=ion_temperature
			loop.density=density
			loop.velocity=velocity

class loopobj(object):
	def __init__(self, index, lines, mags, lengths, loopname=None, frame='heliographic_stonyhurst'):
		#ilo = loopinfo['ila'][index]
		#ihi = loopinfo['ila'][index+1]-1
		#self.field_aligned_coordinate = (loopinfo['looppt_lengths'][ilo:ihi]+0.5)*loopinfo['looplengths'][index]/loopinfo['looplengths_vox'][index]
		#self.coordinate = loopinfo['loopcoords'][ilo:ihi,:]
		#for i in range(0,3): self.coordinate[:,i] *= loopinfo['dvox'][i]
		#self.length = loopinfo['looplengths'][index]
		#self.name = loopinfo['loopnames'][index] # f'loop{index:06d}'
		self.frame = frame
		self.index = index
		#self.field_strength = interpolator(self.coordinate)
		self.field_aligned_coordinate=lengths[index]
		self.coordinate=lines[index]
		self.field_strength=mags[index]
		self.length = np.max(lengths[index])
		if(loopname == None): loopname = f'loop{index:06d}'
		self.name = loopname	
