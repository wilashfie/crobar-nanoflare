# Loop profiles are set using a closed-form approximation to the RTV
# scaling laws: T(l) = T_a*((l/L)*(1-l/L))^g, where L is the loop (whole) length,
# T_a is the apex temperature, and g is an exponent that controls the temperature profile.
# Overall temperature scaling is set so that apex temps are same in both cases. We use the
# apex temp because that drives the heat flux more than the other temperatures. 
# Pressure profile assumes the usual exponential form with the scale height set by the 
# average temperature. We set the pressures so that energies match between scaled and 
# HYDRAD loops. The density is then set by the pressure and temperature profiles. For a given
# temperature profile exponent, the loop is then fully specified, and therefore the radiative
# losses. We also want the radiative loss rates to match those in the EBTEL simulation, so we
# search for the temperature profile exponent that matches the EBTEL losses. This
# last step fully specifies the loop across both corona and TR. If the transition
# region length is desired, it is where T(l)/T_a = c3.

from tqdm.auto import tqdm
#@processify
def compute_loop_profiles(loopindex,loops,hterms,segment_heights,segment_relareas,lbin_cents,time_indices=None):
	
	c2 = 0.9 # Average over apex ratio
	gmax = 20
	ng = 101 # number of scaling exponents to search over
	logtmin = 4
	logtmax = 8
	ntemp = 101
	hfac = 60e8/1.0e6 # Conversion factor from temperature to hydrostatic scale height (cm per Kelvin) 

	rltemp = 10**np.linspace(logtmin,logtmax,ntemp)
	rllogt = np.log10(rltemp)
	rloft = 1.09e-31*rltemp**2*(rllogt <= 4.97) + 8.87e-17/rltemp*(rllogt > 4.97)*(rllogt <= 5.67) + 1.90e-22*(rllogt > 5.67)*(rllogt <= 6.18) + 3.53e-13/rltemp**1.5*(rllogt > 6.18)*(rllogt <= 6.55)+3.46e-25*rltemp**(1.0/3.0)*(rllogt > 6.55)*(rllogt <= 6.9)+5.49e-16/rltemp*(rllogt > 6.9)*(rllogt <= 7.63)+1.96e-27*rltemp**0.5*(rllogt > 7.63)

	loop = loops[loopindex]
	hterm = hterms[loopindex]

	ltot = loop.length # Overall Length
	lengths = ltot*(lbin_cents*ltot)/(ltot) # Arc lengths
	times = loop.time
	hofl = segment_heights[loopindex,:] # Height profile
	aofl = np.clip(segment_relareas[loopindex,:]*area_norms[loopindex],dvox[0]*dvox[1],np.prod(nvox*dvox))# Area profile
	if(time_indices == None): time_indices = np.arange(len(times))
	if(np.isscalar(time_indices)): time_indices = np.array(time_indices,ndmin=1)
	ntimes = np.size(time_indices)
	nl = np.size(lengths)

	gexps = np.insert(10**np.linspace(np.log10(gmax)-0.05*ng,np.log10(gmax),ng-1),0,0)
	tprofs = np.zeros([ng,nl])
	tavg_scals = np.zeros(ng)
	loopvol = np.trapz(aofl,x=lengths)
	rl_scald = np.zeros(ng)
	for i in range(0,ng): 
		tprofs[i,:] = (4*(lengths/ltot)*(1-lengths/ltot))**gexps[i]
		tavg_scals[i] = np.trapz(tprofs[i,:],x=lengths)/ltot
	
	g_out = np.zeros(ntimes,dtype='float32')
	temp_out = np.zeros([ntimes,nl],dtype='float32')
	pres_out = np.zeros([ntimes,nl],dtype='float32')
	pscals = np.zeros([ng,nl],dtype='float32')
	
	for i2 in tqdm(range(ntimes)):
		i = time_indices[i2] 
		tavg = 0.5*(loop.electron_temperature[i]+loop.ion_temperature[i])
		dens = (loop.density[i])
		tmax = np.clip(tavg/c2,10**logtmin,10**logtmax)
		c1 = hterm['c1'][i]
		pres = (dens*tavg)
		#temps = np.clip(tmax*tprofs,10**logtmin,10**logtmax)
		#temps[:,0] = rltemp[0]
		#temps[:,nl-1] = rltemp[0]
		#hscals = tavg*tavg_scals*hfac
		temps = 10**logtmin+(tmax-10**logtmin)*tprofs
		hscals = (10**logtmin+(tmax-10**logtmin)*tavg_scals)*hfac        
		if(np.all(np.isfinite(c1))*np.all(np.isfinite(aofl))*np.isfinite(tavg)*np.all(np.isfinite(dens))*np.isfinite(ltot)*np.isfinite(loopvol)==0):
			print(np.all(np.isfinite(c1)),np.all(np.isfinite(aofl)),tavg,np.all(np.isfinite(dens)),ltot,loopvol)
		rl_ebtel = c1*np.trapz(aofl*np.interp(tavg,rltemp,rloft)*dens**2,x=lengths)*ltot/loopvol
		for j in range(0,ng): 
			pscals[j,:] = np.exp(-hofl/hscals[j])
			pscals[j,:] *= pres*loopvol/np.trapz(pscals[j,:]*aofl,x=lengths)
		dens2s = pscals/temps
		for j in range(0,ng): 
			rl_scald[j] = np.trapz(aofl*np.interp(temps[j,:],rltemp,rloft)*dens2s[j,:]**2,x=lengths)*ltot/loopvol
		g_out[i2] = np.clip(np.interp(rl_ebtel,rl_scald,gexps),0,gmax)
		#temp_out[i2,:] = np.clip(tmax*(4*(lengths/ltot)*(1-lengths/ltot))**g_out[i2],10**logtmin,10**logtmax)
		temp_out[i2,:] = 10**logtmin+(tmax-10**logtmin)*(4*(lengths/ltot)*(1-lengths/ltot))**g_out[i2]
		pres_out[i2,:] = np.exp(-hofl/(hfac*np.trapz(temp_out[i2,:],x=lengths)/ltot))
		pres_out[i2,:] *= pres*loopvol/np.trapz(pres_out[i2,:]*aofl,x=lengths)
	
	return {'temp':temp_out, 'dens':pres_out/temp_out, 'pres':pres_out, 'gexp':g_out}
