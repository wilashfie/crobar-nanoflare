import logging
import warnings
import functools
import time
#import processify
import resource
from tqdm.auto import tqdm

import numpy as np
from numpy.random import Generator, PCG64
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from sunpy.coordinates.frames import Heliocentric
from astropy.utils.console import ProgressBar
from astropy.coordinates import SkyCoord
import astropy.units as u
import pickle
'''
import yt
import numba

import synthesizAR
import synthesizAR.extrapolate
from synthesizAR.util import SpatialPair
from synthesizAR.extrapolate.helpers import from_local, to_local, magnetic_field_to_yt_dataset
'''

# This generates a heating map that's proportional to the field strength plus a footpoint
# heating image (heatmap_2d). The overall heat amount is set to h_rate0.
#@processify.processify
def generate_heatmap_fps(Bfield,heatmap_2d,h_rate0):
	
	Bdomain = np.vstack(np.array((Bfield.domain_left_edge,Bfield.domain_right_edge)))
	Bd_size = Bdomain[1,:]-Bdomain[0,:]
	Bd_ndim = Bfield.domain_dimensions
	Bd_dvox = Bd_size/(Bd_ndim-1.0)
	Bx = np.array(Bfield.r['Bx'].reshape(Bd_ndim))
	By = np.array(Bfield.r['By'].reshape(Bd_ndim))
	Bz = np.array(Bfield.r['Bz'].reshape(Bd_ndim))
	B2 = np.array(Bx*Bx+By*By+Bz*Bz).astype('float32')

	z = np.array(Bfield.r['z'].reshape(Bd_ndim)).astype('float32')
	tot_area = Bd_dvox[0]*Bd_ndim[0]*Bd_dvox[1]*Bd_ndim[1]*u.cm**2

	hotpad_fac = 0.2
	hotpad = np.ceil(hotpad_fac*Bd_ndim).astype('int32')
	tot_area *= (1.0-2*hotpad_fac)**2.0
	slpad = np.index_exp[hotpad[0]:(Bd_ndim[0]-hotpad[0]),hotpad[1]:(Bd_ndim[1]-hotpad[1]),0:(Bd_ndim[2]-hotpad[2])]
	slpad0 = np.index_exp[hotpad[0]:(Bd_ndim[0]-hotpad[0]),hotpad[1]:(Bd_ndim[1]-hotpad[1]),0]
	slpad00 = np.index_exp[hotpad[0]:(Bd_ndim[0]-hotpad[0]),hotpad[1]:(Bd_ndim[1]-hotpad[1])]

	heatmap_3d = 0.0*B2
	heatmap_3d[slpad] = np.exp(-z[slpad]/(200.0e8))*B2[slpad]**0.5
	heatmap_3d[slpad0] += 0.25*heatmap_2d[slpad00]**0.5*np.mean(heatmap_3d[slpad0])/np.mean(heatmap_2d[slpad00]**0.5)
	heatmap_renorm = (h_rate0*tot_area)/np.sum(heatmap_3d*np.prod(Bd_dvox)*u.cm**3).flatten()
	hmx = Bdomain[0,0]+np.arange(Bd_ndim[0])*Bd_dvox[0]
	hmy = Bdomain[0,1]+np.arange(Bd_ndim[1])*Bd_dvox[1]
	hmz = Bdomain[0,2]+np.arange(Bd_ndim[2])*Bd_dvox[2]

	return (heatmap_3d*heatmap_renorm).astype('float32'),(hmx.astype('float32'),hmy.astype('float32'),hmz.astype('float32'))

#@processify.processify
def generate_heatmap(Bfield,heatmap_2d,h_rate0,meanbs,voxmin,nvox,dvox):
	
	z0 = 25.0e8 # nominal height for z-weighting of heat, in cm
	Bdomain = np.vstack(np.array((Bfield.domain_left_edge,Bfield.domain_right_edge)))
	Bd_size = Bdomain[1,:]-Bdomain[0,:]
	Bd_ndim = Bfield.domain_dimensions
	Bd_dvox = Bd_size/(Bd_ndim-1.0)
	hmx = Bdomain[0,0]+(0.5+np.arange(Bd_ndim[0]))*Bd_dvox[0]
	hmy = Bdomain[0,1]+(0.5+np.arange(Bd_ndim[1]))*Bd_dvox[1]
	hmz = Bdomain[0,2]+(0.5+np.arange(Bd_ndim[2]))*Bd_dvox[2]
	Bx = np.array(Bfield.r['Bx'].reshape(Bd_ndim))
	By = np.array(Bfield.r['By'].reshape(Bd_ndim))
	Bz = np.array(Bfield.r['Bz'].reshape(Bd_ndim))
	B2 = np.array(Bx*Bx+By*By+Bz*Bz).astype('float32')

	#meanbx = voxmin[0]+np.arange(nvox[0])*dvox[0]
	#meanby = voxmin[1]+np.arange(nvox[1])*dvox[1]
	#meanbz = voxmin[2]+np.arange(nvox[2])*dvox[2]

	#[hmx2,hmy2,hmz2] = np.indices(Bd_ndim)
	#hmx2 = ((hmx2)*Bd_dvox[0]+Bdomain[0,0]).astype('float32')
	#hmy2 = ((hmy2)*Bd_dvox[1]+Bdomain[0,1]).astype('float32')
	#hmz2 = ((hmz2)*Bd_dvox[2]+Bdomain[0,2]).astype('float32')

	#meanb_interpolator = RegularGridInterpolator((meanbx,meanby,meanbz),meanbs,fill_value=None,bounds_error=False)
	#meanb_interp = meanb_interpolator((hmx2,hmy2,hmz2))

	#z = np.array(Bfield.r['z'].reshape(Bd_ndim)).astype('float32')
	tot_area = Bd_dvox[0]*Bd_ndim[0]*Bd_dvox[1]*Bd_ndim[1]*u.cm**2

	hotpad_fac = 0.2
	hotpad = np.ceil(hotpad_fac*Bd_ndim).astype('int32')
	tot_area *= (1.0-2*hotpad_fac)**2.0
	slpad = np.index_exp[hotpad[0]:(Bd_ndim[0]-hotpad[0]),hotpad[1]:(Bd_ndim[1]-hotpad[1]),0:(Bd_ndim[2]-hotpad[2])]
	slpad0 = np.index_exp[hotpad[0]:(Bd_ndim[0]-hotpad[0]),hotpad[1]:(Bd_ndim[1]-hotpad[1]),0]
	slpad00 = np.index_exp[hotpad[0]:(Bd_ndim[0]-hotpad[0]),hotpad[1]:(Bd_ndim[1]-hotpad[1])]

	heatmap_2d[np.isnan(heatmap_2d)] = 0

	#heatmap_3d = 0.0*meanb_interp
	#heatmap_3d[slpad] = meanb_interp[slpad]
	#hmw_norm = np.sum(meanb_interp,axis=2)
	heatmap_3d = 0.0*B2
	##heatmap_3d[slpad] = (B2[slpad]**1.0)*np.exp(-z[slpad]/z0) # This one includes a height weighting
	##hmw_norm = np.sum((B2**1.0)*np.exp(-z/z0),axis=2)
	heatmap_3d[slpad] = B2[slpad]**0.33 # This one doesn't
	hmw_norm = np.sum(B2**0.33,axis=2)
	for i in range(0,Bd_ndim[2]): heatmap_3d[:,:,i] *= (heatmap_2d)/hmw_norm
	#heatmap_3d[slpad] = np.exp(-z[slpad]/(200.0e8))*B2[slpad]**0.5
	#heatmap_3d[slpad0] += 0.25*heatmap_2d[slpad00]**0.5*np.mean(heatmap_3d[slpad0])/np.mean(heatmap_2d[slpad00]**0.5)
	heatmap_renorm = (h_rate0*tot_area)/np.sum(heatmap_3d*np.prod(Bd_dvox)*u.cm**3).flatten()

	#return heatmap_3d*heatmap_renorm,(hmx,hmy,hmz)
	return (heatmap_3d/Bd_dvox[2]).astype('float32'),(hmx.astype('float32'),hmy.astype('float32'),hmz.astype('float32'))

# This routine generates the locations of heating events, as well as their overall number.
# the 3D array heatmap defines the amount of heating per unit volume, while meane is the
# average event energy which defines the number of events (event energy distribution assumed
# to be independent of position). dvox is the size of the grid cells. Locations are returned
# as grid cell indices (i.e., using the voxel grid of heatmap).
def generate_heating_locations(heatmap,dvox,meane):

	[dx,dy,dz]=[1,1,1]
	[xa,ya,za] = np.indices(heatmap.shape)
	hm_tot = np.sum(heatmap)
	hm_flat = heatmap.flatten()/hm_tot
	xa_flat = xa.flatten()
	ya_flat = ya.flatten()
	za_flat = za.flatten()
	indices_sort = np.argsort(hm_flat)
	xa_sort = xa_flat[indices_sort]
	ya_sort = ya_flat[indices_sort]
	za_sort = za_flat[indices_sort]
	mag_sort = hm_flat[indices_sort]
	mag_weight = abs(mag_sort).astype('float64').cumsum()
	mag_cdf = mag_weight/mag_weight.max()
	mag_indices = np.linspace(0,len(mag_cdf)-1,len(mag_cdf)).round().astype(np.int64)
	
	number_events = np.random.poisson(np.prod(dvox)*u.cm**3*hm_tot/(meane*u.erg))

	event_indices = np.interp(np.random.uniform(0,1,number_events),mag_cdf,mag_indices).round().astype(np.int64)
	seed_points = np.zeros([3,number_events],dtype='float32')
	seed_points[0,:]=xa_sort[event_indices]
	seed_points[1,:]=ya_sort[event_indices]
	seed_points[2,:]=za_sort[event_indices]

	return seed_points

# Generate an event energy CDF based on a powerlaw distribution
# (i.e., the number vs. energy distribution is a power law).
def get_cdf(heating_options):
	nnrg = 40000
	index = heating_options['index']
	emin = np.dtype('float64').type((heating_options['emin']/u.erg).value)
	emax = np.dtype('float64').type((heating_options['emax']/u.erg).value)
	loge = np.linspace(np.log(emin),np.log(emax),nnrg,dtype='float64')
	loge2 = loge-np.log(emin)
	dloge = loge[1]-loge[0]
	pdf_fac1 = np.exp(loge2*(index+1.0))
	pdf_fac2 = emin**(index+1.0)
	cdf = np.cumsum(dloge*pdf_fac1)*pdf_fac2
	pdf_fac3 = 1.0/np.max(cdf)
	meane = np.sum(dloge*np.exp(loge2*(index+2)))*pdf_fac2*pdf_fac3*emin
	cdf = cdf*pdf_fac3
	cdf = np.concatenate((np.array([0]),cdf[0:-1]))
		
	return loge, cdf, meane

# Generate random samples with a given probability distribution using
# the provided CDF, by interpolating:
def eval_cdf(loge,cdf,n2,rg=None):
	if(rg==None): rg = Generator(PCG64())
	##return np.exp(np.interp(np.random.uniform(0,1,n2),cdf,loge))
	#return np.exp(np.interp(rg.uniform(0,1,n2),cdf,loge))
	return np.exp(np.interp(rg.random(size=n2,dtype='float32'),cdf,loge))

# Heating is divided into chunks, to reduce memory usage and
# memory usage creep due to fragmentation in the python memory
# manager. This routine computes the voxel cell range for chunking:
def get_chunk_range(ichunk, nchunk_xyz, voxmin, nvox, dvox):

	chunksize = np.ceil(nvox/nchunk_xyz).astype('int32')
	ijkchunk = np.array(np.unravel_index(ichunk,nchunk_xyz))
	ijklo = ijkchunk*chunksize
	ijkhi = np.clip(ijklo+chunksize,0,nvox)
	chunksize2=ijkhi-ijklo
	[chunkx,chunky,chunkz] = np.indices(chunksize2)
	chunkx = ((chunkx+ijklo[0])*dvox[0]+voxmin[0]).astype('float32')
	chunky = ((chunky+ijklo[1])*dvox[1]+voxmin[1]).astype('float32')
	chunkz = ((chunkz+ijklo[2])*dvox[2]+voxmin[2]).astype('float32')

	return chunkx,chunky,chunkz,ijklo,ijkhi

def get_events_original(heatmap, ijkoffset, duration, t_tot, dt, cdf, loge, area_voxels, voxmin, dvox,
		vox_ioffs,kflat,kernones,loopids,nloops,nt,nloops_chunk,loops_chunk, mag_scal, ev_rad, meane):

	nvox = np.array(loopids.shape)
	locs = generate_heating_locations(heatmap,dvox,meane)
	n2 = locs[0,:].size
	t0s = np.round(np.random.uniform(duration/u.s, (t_tot-duration)/u.s-1, n2)/dt).value.astype('int32')
	mags = np.array(((eval_cdf(loge,cdf,n2)/(0.5*duration)).value/area_voxels).astype('float32'))

	return mags,t0s,locs

def get_events_voxelwise(heatmap, ijkoffset, duration, t_tot, dt, cdf, loge, area_voxels, voxmin, dvox,
		vox_ioffs,kflat,kernones,loopids,nloops,nt,nloops_chunk,loops_chunk, mag_scal, ev_rad, meane):

	rg = Generator(PCG64())
	nvox = np.array(heatmap.shape)
	nx = nvox[0]
	ny = nvox[1]
	nz = nvox[2]
	voxvol = np.prod(dvox)
	hmv = (voxvol*u.cm**3*heatmap).to(u.erg).value
	magfac = 2.0/duration.value/area_voxels
	tmin = (duration/u.s).value
	tmax = ((t_tot-duration)/u.s-1).value
	dtval = dt.value

	energypartition = 0.9
	energymap = meane*hmv**energypartition/np.max(hmv**energypartition)
	numbermap = hmv**(1.0-energypartition)*np.max(hmv**energypartition)/meane
	
	#numbermap[np.where(energymap < 1.0e-5*meane)] = 0.0
	
	#print(np.min(numbermap),np.min(hmv),np.min(heatmap))

	[ix,iy,iz] = np.indices(nvox,dtype='int32')
	number_events = rg.poisson(numbermap) #0.10+np.zeros(nvox,dtype='uint16'))
	#number_events = np.random.poisson(0.05+np.zeros(nvox,dtype='uint16'))
	wnz = np.where(number_events > 0)
	ntot = np.sum(number_events)
	nlocs = np.sum(number_events > 0)
	#print("Approximate total number of events = ",ntot,',',nlocs,'unique locations')
	locs_out = np.zeros((ntot,3),dtype='int32')
	mags_out = np.zeros(ntot,dtype='float32')
	times_out = np.zeros(ntot,dtype='float32')
	count=0
	for index in range(0,nlocs):
		#print(index)
		[i,j,k] = [wnz[0][index],wnz[1][index],wnz[2][index]]
		n = number_events[i,j,k]
		#if((index % (nlocs/10).astype('int32'))==0): 
		#	print(index,i,j,k,n,count)
		ihi=count+n
		locs_out[count:ihi,:] = np.tile([i,j,k],[n,1])
		#times_out[count:ihi] = np.random.uniform(tmin,tmax,n)
		times_out[count:ihi] = tmin+tmax*rg.random(size=n,dtype='float32')
		#mags_out[count:ihi] = eval_cdf(loge,cdf,n,rg=rg)*(hmv[i,j,k]/meane)/0.10
		mags_out[count:ihi] = eval_cdf(loge,cdf,n,rg=rg)*(energymap[i,j,k])/meane
		count+=n

	#print(count)

#	for i in range(0,nx):
#		for j in range(0,ny):
#			for k in range(0,nz):
#				#number_events = np.random.poisson(np.prod(dvox)*u.cm**3*heatmap[i,j,k]/(meane*u.erg))
#				number_events = np.random.poisson(0.05)#hmv[i,j,k]/meane)
#				if(number_events > 0):
#					mags = eval_cdf(loge,cdf,number_events)*(hmv[i,j,k]/meane)/0.05
#					#mags = np.array(((eval_cdf(loge,cdf,number_events)/(0.5*duration)).value/area_voxels).astype('float32'))
#					times = np.random.uniform(tmin, tmax, number_events)
#					#times = np.round(np.random.uniform(duration/u.s, (t_tot-duration)/u.s-1, number_events)/dt).value.astype('int32')
#					locs = np.tile([i,j,k],[number_events,1])
#					if(initialized == 0):
#						mags_out = mags
#						times_out = times
#						locs_out = locs
#						initialized = 1
#					if(initialized == 1):
#						mags_out = np.hstack((mags_out,mags))
#						times_out = np.hstack((times_out,times))
#						locs_out = np.vstack((locs_out,locs))

	return (mags_out*magfac).astype('float32'),np.round(times_out/dtval).astype('int32'),locs_out.transpose()

# This routine generates heating events and applies the heating to a chunk of voxels:
#@processify.processify
def heat_chunk(heatmap, ijkoffset, duration, t_tot, dt, cdf, loge, area_voxels, voxmin, dvox,
		vox_ioffs,kflat,kernones,loopids,nloops,nt,nloops_chunk,loops_chunk, mag_scal, ev_rad, meane):
	


	t0 = time.time()

	from scipy.sparse import csr_matrix
	from scipy.sparse import csc_matrix
	from scipy.sparse import coo_matrix

	#sz_voxels = 2*ev_rad+1	
	#nvox = np.array(loopids.shape)
	#heatmap_sz = np.array(heatmap.shape)
	#locs = generate_heating_locations(heatmap,dvox,meane)
	#for j in range(0,3): locs[j,:] += ijkoffset[j]-ev_rad[j] # np.floor(0.5*sz_voxels[j])
	#locs = np.floor(locs).astype('int32')
	#n2 = locs[0,:].size
	#t0s = np.round(np.random.uniform(duration/u.s, (t_tot-duration)/u.s-1, n2)/dt).value.astype('int32')
	#mags = np.array(((eval_cdf(loge,cdf,n2)/(0.5*duration)).value/area_voxels).astype('float32'))
	#vox_ioff2 = np.expand_dims(vox_ioffs,1)
	#nve = np.expand_dims(nvox,1)

	sz_voxels = 2*ev_rad+1	
	nvox = np.array(loopids.shape)
	heatmap_sz = np.array(heatmap.shape)
	##locs = generate_heating_locations(heatmap,dvox,meane)
	[mags,t0s,locs] = get_events_voxelwise(heatmap, ijkoffset, duration, t_tot, dt, cdf, loge, area_voxels, voxmin, dvox,
		vox_ioffs,kflat,kernones,loopids,nloops,nt,nloops_chunk,loops_chunk, mag_scal, ev_rad, meane)
	##print(locs.shape)
	#[mags,t0s,locs] = get_events_original(heatmap, ijkoffset, duration, t_tot, dt, cdf, loge, area_voxels, voxmin, dvox,
	#	vox_ioffs,kflat,kernones,loopids,nloops,nt,nloops_chunk,loops_chunk, mag_scal, ev_rad, meane)
	##print(ijkoffset,ev_rad)
	for j in range(0,3): locs[j,:] += ijkoffset[j]-ev_rad[j] # np.floor(0.5*sz_voxels[j])
	locs = np.floor(locs).astype('int32')
	n2 = locs[0,:].size
	#t0s = np.round(np.random.uniform(duration/u.s, (t_tot-duration)/u.s-1, n2)/dt).value.astype('int32')
	#mags = np.array(((eval_cdf(loge,cdf,n2)/(0.5*duration)).value/area_voxels).astype('float32'))
	vox_ioff2 = np.expand_dims(vox_ioffs,1)
	nve = np.expand_dims(nvox,1)


	nheats = (nloops_chunk*nt)
	heating = np.zeros(nloops_chunk*nt,dtype='float64')
	loop_nevents = np.zeros(nloops_chunk*nt,dtype='uint32')

	t0 = t0-time.time()
	t1 = time.time()

	heatmap_out = np.zeros(nvox,dtype='float32')

	#print("Checkpoint 1")

	if((np.sum(ijkoffset < 0) > 0) or (np.sum((nvox-heatmap_sz) < np.ceil(0.5*sz_voxels)) > 0)):
		for i in range(0,kflat.size):
			locs2 = locs+vox_ioff2[:,:,i]
			locs2[2,:] = np.abs(locs2[2,:])
			bflg = np.array(np.where(np.prod((locs2 > -1)*(locs2 < nve),axis=0))).flatten()
			#print('heat_chunk diagnostic:',locs[0,0].value)
			#test = np.bincount(loops_chunk[loopids[locs2[0,bflg],locs2[1,bflg],locs2[2,bflg]]]*nt+t0s[bflg],weights=mags[bflg]*kflat[i],minlength=nheats)
			#print(test[0],test.shape,heating[0])
			heating += np.bincount(loops_chunk[loopids[locs2[0,bflg],locs2[1,bflg],locs2[2,bflg]]]*nt+t0s[bflg],weights=mags[bflg]*kflat[i],minlength=nheats)
			loop_nevents += np.bincount(loops_chunk[loopids[locs2[0,bflg],locs2[1,bflg],locs2[2,bflg]]]*nt+t0s[bflg],minlength=nheats).astype('uint32')
			np.add.at(heatmap_out,(locs2[0,bflg],locs2[1,bflg],locs2[2,bflg]),mags[bflg]*kflat[i])
	else:
		for i in range(0,kflat.size):
			locs2 = locs+vox_ioff2[:,:,i]
			heating += np.bincount(loops_chunk[loopids[tuple(locs2)]]*nt+t0s,weights=mags*kflat[i],minlength=nheats)
			loop_nevents += np.bincount(loops_chunk[loopids[tuple(locs2)]]*nt+t0s,minlength=nheats).astype('uint32')
			np.add.at(heatmap_out,tuple(locs2),mags*kflat[i])

	#print("Checkpoint 2")

	t1 = t1-time.time()
	return [np.reshape(heating,(nloops_chunk,nt)),n2,t0,t1,heatmap_out,np.reshape(loop_nevents,(nloops_chunk,nt))]

class volumetric_poisson_powerlaw_nanoflares(object):

	def __init__(self,loopids,loopid_info):
		self.nv = loopid_info['nvox']
		self.vmin = loopid_info['voxmin']
		self.dv = loopid_info['dvox']
		self.nl = len(loopid_info['looplengths'])
		self.loopids = loopids
		self.loopid_info = loopid_info
		self.loop_closed = loopid_info['loop_closed']
		self.loopnames = np.array(loopid_info['loopnames'])
		#self.loop_volumes = loop_volumes # np.zeros(self.nl+1)
		self.loop_volumes = loopid_info['loop_volumes']
		#self.heatmap2 = np.zeros(self.nv,dtype='float32')
		#for i in range(0,self.nv[2]): np.add.at(self.loop_volumes,loopids[:,:,i],1)
		#self.loop_volumes *= np.prod(self.dv)
		self.inited=1

	def get_kernel(self,heating_options):
		size0 = np.array(heating_options['event_size'])
		dvox = self.loopid_info['dvox']
		ev_rad = np.round(np.array(0.5*size0/dvox)).astype('int32')
		sz_voxels = 2*ev_rad+1 # np.round(np.array(size0/dvox)).astype('int32')+1
		area_voxels = np.prod(sz_voxels)
		[vox_ixa,vox_iya,vox_iza] = np.indices(sz_voxels,dtype='int32')
		vox_ixa0 = (vox_ixa.flatten())
		vox_iya0 = (vox_iya.flatten())
		vox_iza0 = (vox_iza.flatten())
		event_nvox = len(vox_ixa0)
		vox_kernrad = 2.0*(((vox_ixa0-0.5*(sz_voxels[0]-1))/(sz_voxels[0]-1))**2+((vox_iya0-0.5*(sz_voxels[1]-1))/(sz_voxels[1]-1))**2+((vox_iza0-0.5*(sz_voxels[2]-1))/(sz_voxels[2]-1))**2)**0.5
		kernel = (1.0/6.0-(vox_kernrad**2/2.0-vox_kernrad**3/3.0))*(vox_kernrad < 1.0)
		kernel = kernel*area_voxels/np.sum(kernel)
		kflat = kernel.flatten()
		wkern = np.where(kernel > 1.0e-3/event_nvox)
		kflat = kflat[wkern].astype('float32')
		vox_ixa0 = vox_ixa0[wkern]
		vox_iya0 = vox_iya0[wkern]
		vox_iza0 = vox_iza0[wkern]
		vox_ioffs = np.vstack([vox_ixa0,vox_iya0,vox_iza0]).astype('int32')
		event_nvox = len(vox_ixa0)
		kernones = 1+np.zeros(event_nvox,dtype='int32')
		return vox_ioffs,kflat,ev_rad,area_voxels,kernones

	def generate_events(self,heating_options):
		duration = heating_options['duration']		
		t_tot = heating_options['t_tot']
		nt = heating_options['nt']
		self.times = np.linspace(0,t_tot/u.s,nt)
		[self.loop_heating,self.loop_nevents] = self.compute_events(heating_options)
		self.loop_heating[0:self.nl,0] += 1.0e-08*self.loop_volumes[0:self.nl]
		self.duration=duration

	#@processify.processify
	def compute_events(self,heating_options):
		
		import os
		import pickle
		#h_rate0 = heating_options['h_rate']/u.erg
		size0 = np.array(heating_options['event_size'])
		duration = heating_options['duration']		
		t_tot = heating_options['t_tot']
		nt = heating_options['nt']
		#self.times = np.linspace(0,t_tot/u.s,nt)
		dt = self.times[1]-self.times[0]

		# Need to put the total heating rate/total area back in here...

		[loge,cdf,meane] = get_cdf(heating_options)
		#number_events = np.random.poisson(h_rate0*t_tot*tot_area/meane)
		[vox_ioffs,kflat,ev_rad,area_voxels,kernones] = self.get_kernel(heating_options)
		
		heatmap = heating_options['heatmap']
		hmxyz = heating_options['hmxyz']
		#[heatmap_name,heatmap,hmxyz,hmimg] = pickle.load(open(heating_options['heating_distribution_name'],"rb"))
		#[heatmap_name,heatmap,hmxyz,hmimg] = load_heating_distribution(heating_options['heating_distribution_name'])
		heatmap *= heating_options['heatmap_factor']	
		heatmap_interp = RegularGridInterpolator(hmxyz,heatmap,fill_value=None,bounds_error=False)
		#magscal_interp = RegularGridInterpolator(heating_options['hmxyz'],heating_options['magscal'],fill_value=None,bounds_error=False)

		nchunk_xyz = np.array([20,20,20],dtype='int32')
		nchunk_total = np.prod(nchunk_xyz).astype('uint32')
		chunk_xyzlo = np.indices(nchunk_xyz,dtype='float32')
		chunk_xyzhi = chunk_xyzlo+1
		for i in range(0,3):
			chunk_xyzlo[i] = chunk_xyzlo[i]/nchunk_xyz[i]
			chunk_xyzhi[i] = chunk_xyzhi[i]/nchunk_xyz[i]

		loop_heating = np.zeros((self.nl+1,nt),dtype='float32')
		loop_nevents = np.zeros((self.nl+1,nt),dtype='uint32')
		tstart = time.time()
		chunk_loop_flg = np.zeros(self.nl+1,dtype='bool')
		loops_chunk = np.zeros(self.nl+1,dtype='uint32')
		#print('Generating approximately',number_events,' events, ',len(kflat),'voxels per event')
		for i in tqdm(range(0,nchunk_total), desc='Generating heating events', unit='chunk'):
			t0start = time.time()
			[chunkx,chunky,chunkz,ijklo,ijkhi] = get_chunk_range(i, nchunk_xyz, self.vmin, self.nv, self.dv)
			vox_ijklo = np.clip(ijklo-ev_rad,0,self.nv).astype('int32')
			vox_ijkhi = np.clip(ijkhi+ev_rad,0,self.nv).astype('int32')
			ijkoffset = ijklo-vox_ijklo
			#meanb = heating_options['meanbs'][ijklo[0]:ijkhi[0],ijklo[1]:ijkhi[1],ijklo[2]:ijkhi[2]]
			chunkheat = np.abs(heatmap_interp((chunkx,chunky,chunkz)))*u.erg*t_tot/u.s/u.cm**3
			chunk_magscal = 0 # magscal_interp((chunkx,chunky,chunkz))*ms_fac
			if(np.sum(chunkheat)*np.prod(self.dv) > 0):
				voxmin2 = vox_ijklo*self.dv+self.vmin
				chunk_lids = self.loopids[vox_ijklo[0]:vox_ijkhi[0],vox_ijklo[1]:vox_ijkhi[1],vox_ijklo[2]:vox_ijkhi[2]]
				chunk_loop_flg[:]=False
				loops_chunk[:]=0
				chunk_loop_flg[chunk_lids]=True
				nloops_chunk=np.sum(chunk_loop_flg)
				loops_chunk[chunk_loop_flg]=np.arange(nloops_chunk,dtype='uint32')
				[heating,n2,t0,t1,heatmap_out,loop_nevents_chunk] = heat_chunk(chunkheat, 
						ijkoffset, duration, t_tot, dt, cdf,loge, area_voxels, voxmin2, 
						self.dv, vox_ioffs, kflat, kernones, chunk_lids, self.nl,
						nt, nloops_chunk, loops_chunk, chunk_magscal, ev_rad, meane)
				loop_heating[chunk_loop_flg,:] += heating
				loop_nevents[chunk_loop_flg,:] += loop_nevents_chunk
				#self.heatmap2[vox_ijklo[0]:vox_ijkhi[0],vox_ijklo[1]:vox_ijkhi[1],vox_ijklo[2]:vox_ijkhi[2]] += heatmap_out
			if((i % (nchunk_total/20).astype('int32'))==0): 
					print('Done with chunk ',i,' out of ',nchunk_total,', ',time.time()-tstart,' seconds elapsed')
		print('Done generating events; total time = ',time.time()-tstart)

		#print(h_rate0,0.5*duration*np.sum(self.heatmap2)/(t_tot*tot_area))

		return loop_heating, loop_nevents

		#self.loop_heating = loop_heating
		#self.loop_nevents = loop_nevents

		#self.loop_heating[:,0] += 1.0e-08*self.loop_volumes
		#self.duration=duration

	def create_events_dict(self):
		heatdict = {
			'duration': self.duration.value,  
			'duration_unit': self.duration.unit.to_string(),
			'loopnames': self.loopnames,
			'closed': self.loop_closed,
			'loop_volumes': self.loop_volumes,
			'heating': self.loop_heating,
			'times': self.times.value
		}
		return heatdict

	def restore_events(self, heatdict):
		self.duration = heatdict['duration']*u.s
		self.loopnames = heatdict['loopnames']
		self.loop_closed = heatdict['closed']
		self.loop_volumes = heatdict['loop_volumes']
		self.loop_heating = heatdict['heating']
		self.times = heatdict['times']
		

	def calculate_event_properties(self,loop):
		
		vol_fid = 10.0e24
		lfrac = 1.0 # loop.full_length.to(u.cm).value/100.0e8
		magmax = 100.0e0
		thalf = 0.5*self.duration.value
		loopindex = np.where(self.loopnames == loop.name)
		bfrac = 1.0 # (lfrac*self.loop_meanfields[loopindex]/50.0)**0.5
		open_fac = 1.0*self.loop_closed[loopindex[0]]+2.0*(self.loop_closed[loopindex[0]] == 0)
		open_fac = open_fac[0]
		length = (self.loopid_info['looplengths'][loopindex][0])
		meanarea = self.loop_volumes[loopindex]/(length*u.cm)
		magnitude = np.clip((self.loop_heating[loopindex,:].flatten())*lfrac*bfrac/self.loop_volumes[loopindex],0,magmax)*u.erg/open_fac
		#magnitude = np.clip((self.loop_heating[loopindex,:].flatten())*bfrac/vol_fid/lfrac,0,magmax)*u.erg/open_fac
		loop_length = 1.0*open_fac*length/2.0
		magnitude[0]=0
		rise_start = self.times[magnitude > 0]*u.s
		rise_start += np.random.uniform(self.times[0],self.times[1],len(rise_start))*u.s
		rise_end = rise_start+thalf*u.s
		decay_start = rise_start+thalf*u.s
		decay_end = rise_start+2*thalf*u.s
        
		return{'magnitude':magnitude[magnitude > 0],'rise_start':rise_start,'rise_end':rise_end,'decay_start':decay_start,'decay_end':decay_end, 'loop_length':loop_length}
	

	def calculate_uniform_event_properties(self,loop):
		dt = (self.times[1]-self.times[0])
		loopindex = np.where(self.loopnames == loop.name)

		lfrac = 1.0
		bfrac = 1.0
		magmax = 100.0e0

		rise_start = self.times[0:-1]*u.s
		rise_end = rise_start + dt*u.s  # Extend the rise time
		decay_start = rise_start + dt*u.s  # Start decay at extended rise end
		decay_end = rise_start + 2*dt*u.s  # End decay later to create overlap

		open_fac = 1.0*self.loop_closed[loopindex[0]]+2.0*(self.loop_closed[loopindex[0]] == 0)
		open_fac = open_fac[0]
		length = (self.loopid_info['looplengths'][loopindex][0])
		loop_length = 1.0*open_fac*length/2.0

		magnitude = np.clip((self.loop_heating[loopindex,:].flatten()[:-1])*lfrac*bfrac/self.loop_volumes[loopindex],0,magmax)*u.erg/open_fac

		e_tot = np.sum(magnitude)*self.duration.value/2
		mag = e_tot/self.times[-1]*np.ones(len(self.times)-1)
		mag[0] = 0.0 # No heating at the first time step?


		#magnitude = mag*np.ones(len(self.times)-1)*u.erg/open_fac
		#magnitude = mag*np.ones(len(self.times))*u.erg/open_fac


		return{'magnitude':mag[mag>0],'rise_start':rise_start,'rise_end':rise_end,'decay_start':decay_start,'decay_end':decay_end, 'loop_length':loop_length}

