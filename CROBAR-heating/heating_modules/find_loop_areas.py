# This routine calculates the relative area of a loop as a function of it's (relative)
# length. We also compute the heights at the same time This is done by counting the 
# number of voxels in the loop as a function of its length, which ensures a close 
# match with the computed loop area/volume and its volume as implemented in the voxel grid:

from tqdm.auto import tqdm
def find_loop_areas(loopids,loopid_info,voxlengths):

	import numpy as np

	# Unpack some values from the loopid_info dictionary:
	nvox = loopid_info['nvox']
	nloops = len(loopid_info['looplengths'])
	lmax = loopid_info['lmax_scal']
	dvox = loopid_info['dvox']

	# First, we compute the areas on a coarse grid to reduce
	# number count issues for short loops containing few voxels. 
	# This computes the bin edges and centers, which will later
	# be used to reinterpolate to higher resolution: 
	nlbins = 15
	lbin_edges = np.linspace(0,1,num=nlbins+1)
	lbin_cents = 0.5*(lbin_edges[0:nlbins]+lbin_edges[1:nlbins+1])
	dlbins = lbin_edges[1]-lbin_edges[0]

	# Now we accumulate the number of voxels in each length bin
	# for each loop. To reduce memory usage, we do it one z slice
	# at a time. We also accumulate the heights in a similar
	# fashion.
	segment_volumes = np.zeros([nloops+1,nlbins],dtype='float64')
	segment_heights0 = np.zeros([nloops+1,nlbins],dtype='float64')
	for i in tqdm(range(0,nvox[2]), desc='Finding loop areas'):
		segment_ids = np.clip(np.floor(voxlengths[:,:,i]/dlbins/lmax),0,nlbins-1).astype('uint8')
		np.add.at(segment_heights0,[loopids[:,:,i],segment_ids],(i+0.5)*dvox[2])
		np.add.at(segment_volumes,[loopids[:,:,i],segment_ids],1)
	loopvols = np.sum(segment_volumes,axis=1)
	segment_volumes = np.clip(segment_volumes,1,None) # Do this so there are no zero-area segments
	segment_heights0 /= segment_volumes # Convert segment summed heights to segment average heights

	# Trim the nloops+1 entry from the array (nloops+1 is used for voxels with no loop)
	segment_volumes = segment_volumes[0:nloops,:] 
	segment_heights0 = segment_heights0[0:nloops,:]
	
	loop_amaxes = np.clip(np.max(segment_volumes,axis=1),1,None)

	# Now we set up the arrays for interpolation of the coarse-grid
	# to the fine grid:
	nlbins2 = 255
	lbin_edges2 = np.linspace(0,1,num=nlbins2+1)
	lbin_cents2 = 0.5*(lbin_edges2[0:nlbins2]+lbin_edges2[1:nlbins2+1])
	dlbins2 = lbin_edges2[1]-lbin_edges2[0]

	from scipy.interpolate import UnivariateSpline
	from scipy.interpolate import interp1d as interp1d
	segment_relareas = np.zeros([nloops+1,nlbins2],dtype='float32')
	segment_heights = np.zeros([nloops+1,nlbins2],dtype='float32')
	for i in tqdm(range(0,nloops), desc='Interpolating loop areas'): 
		segment_relareas[i,:] = np.clip(1/interp1d(lbin_cents,1/segment_volumes[i,:],fill_value="extrapolate")(lbin_cents2),1,loop_amaxes[i])/loop_amaxes[i]
		segment_heights[i,:] = np.clip(interp1d(lbin_cents,segment_heights0[i,:],fill_value="extrapolate")(lbin_cents2),0.5*dvox[2],(nvox[2]*dvox[2]))

	area_norms = loopvols[0:nloops]*np.prod(dvox)/loopid_info['looplengths'][0:nloops]/loop_amaxes

	return lbin_cents2, segment_relareas, segment_heights, loop_amaxes, area_norms
