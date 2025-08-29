import numpy as np
from load_terms import load_terms
from tqdm.auto import tqdm

'''
class general_profile(object):
	# Prototype for example purposes...
	def __init__(self, ebtel_profiles, hydrad_profiles):
		self.ebtel_profiles = ebtel_profiles
		self.hydrad_profiles = hydrad_profiles
		
	def calculate(self, loop_index, time_index):
		profile = self.hydrad_profiles.calculate(loop_index, time_index)		
		if(profile is None):
			profile = self.ebtel_profiles.calculate(loop_index, time_index)
			
		return profile
'''

def ebtel_powerlaw_radloss(logt):
	temp = 10**logt
	return (1.09e-31*temp**2*(logt <= 4.97) + 8.87e-17/temp*(logt > 4.97)*(logt <= 5.67) + 
			1.90e-22*(logt > 5.67)*(logt <= 6.18) + 3.53e-13/temp**1.5*(logt > 6.18)*(logt <= 6.55) +
			3.46e-25*temp**(1.0/3.0)*(logt > 6.55)*(logt <= 6.9) + 
			5.49e-16/temp*(logt > 6.9)*(logt <= 7.63) + 1.96e-27*temp**0.5*(logt > 7.63))

class variable_profile(object):
	def __init__(self, nvox, dvox, times, loops, area_config, 
				c2 = 0.9, gmax=10, ng=41, rltemp=None, rloft=None, hfac = 60e8/1.0e6):

		loop_amaxes = area_config['loop_amaxes']; segment_relareas = area_config['segment_relareas']
		self.times=times; self.c2=c2; self.gmax=gmax; self.ng=ng
		self.hofls=area_config['segment_heights']; self.hfac=hfac; self.lfracs=area_config['lbin_cents']

		if(rltemp is None):	rltemp = 10**np.linspace(4.5,7.5,61)
		self.tmin, self.tmax, self.ntemp = np.min(rltemp), np.max(rltemp), len(rltemp)
		self.logtmin, self.logtmax = np.log10(self.tmin), np.log10(self.tmax)
		self.rltemp = rltemp; self.rllogt = np.log10(self.rltemp)
		
		if(rloft is None): rloft = ebtel_powerlaw_radloss(self.rllogt)
		self.rloft = rloft

		self.nleng = len(self.lfracs)
		self.nloop = len(loops)
		self.ntime = len(times)
		
		self.ltots = np.zeros(self.nloop, dtype=np.float32)
		self.lengs = np.zeros([self.nloop, self.nleng], dtype=np.float32)
		self.vols = np.zeros([self.nloop, self.nleng], dtype=np.float32)
		self.loopvols = np.zeros(self.nloop, dtype=np.float32)
		for i in tqdm(np.arange(0,self.nloop)):
			self.ltots[i] = loops[i].length
			self.lengs[i] = self.lfracs*self.ltots[i]
			self.vols[i] = segment_relareas[i]*loop_amaxes[i]
			self.loopvols[i] = np.sum(self.vols[i])
		
		self.gexps = np.linspace(0, self.gmax, self.ng)
		self.tprofs = np.zeros([self.ng, self.nleng])
		self.tavg_scals = np.zeros([self.nloop, self.ng])
		for i in range(0,ng): 
			self.tprofs[i] = (4*self.lfracs*(1-self.lfracs))**self.gexps[i]
			for j in range(0,self.nloop): self.tavg_scals[j,i] = np.trapezoid(self.tprofs[i],x=self.lengs[i])/self.ltots[i]

			
		self.tavgs = np.zeros([self.nloop, self.ntime],dtype=np.float32)
		self.denss = np.zeros([self.nloop, self.ntime],dtype=np.float32)
		self.tmaxs = np.zeros([self.nloop, self.ntime],dtype=np.float32)
		self.trats = np.zeros([self.nloop, self.ntime],dtype=np.float32)
		self.press = np.zeros([self.nloop, self.ntime],dtype=np.float32)
		for i in tqdm(np.arange(0,self.nloop)):
			loopt = loops[i].time
			self.tavgs[i] = np.interp(times, loopt, 0.5*(loops[i].electron_temperature+loops[i].ion_temperature))
			self.denss[i] = np.interp(times, loopt, loops[i].density)
			self.trats[i] = np.interp(times, loopt, load_terms(loops[i])['c1'])
			self.tmaxs[i] = np.clip(self.tavgs[i]/self.c2, self.tmin, self.tmax)
			self.press[i] = self.denss[i]*self.tavgs[i]
			
	def calculate(self, loop_index, time_index):
		l,t = loop_index, time_index
		temps = np.clip(self.tmaxs[l,t]*self.tprofs,self.tmin,self.tmax)
		hscals = self.tavgs[l,t]*self.hfac
		rl_ebtel = (1+self.trats[l,t])*np.sum(self.vols[l]*np.interp(self.tavgs[l,t],self.rltemp,self.rloft))*self.denss[l,t]**2
		pscals = np.zeros([self.ng,self.nleng])
		for j in range(0,self.ng):
			pscals[j,:] = np.exp(-self.hofls[l]/hscals)
			pscals[j,:] *= np.sum(self.press[l,t]*self.loopvols[l])/np.sum(pscals[j]*self.vols[l])
		dens2s = pscals/temps
		rl_scald = np.zeros(self.ng)
		for j in range(0,self.ng):
			rl_scald[j] = np.sum(self.vols[l]*np.interp(temps[j],self.rltemp,self.rloft)*dens2s[j]**2)
		g_out = np.clip(np.interp(rl_ebtel, rl_scald, self.gexps), 0, self.gmax)
		temp_out = np.clip(self.tmaxs[l,t]*(4*self.lfracs*(1-self.lfracs))**g_out,self.tmin,self.tmax)
		pres_out = np.exp(-self.hofls[l]/(self.hfac*np.trapezoid(temp_out,x=self.lengs[l])/self.ltots[l]))
		pres_out *= self.press[l,t]*self.loopvols[l]/np.sum(pres_out*self.vols[l])
		
		return temp_out, pres_out/temp_out, pres_out, g_out
