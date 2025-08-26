import time, sunpy, resource, astropy.units as u, numpy as np, scipy.ndimage as ndimage
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicStonyhurst, Heliocentric
from scipy.sparse import diags, csr_matrix, csc_matrix
from reconstruct_2comp_3Dloops import get_psfmat
from local_cartesian_transform import transform, vox2pix, wrld2pix


def general_fwdmat(map_in, loopid_info, voxmap, bin_fac=3, psf_size_px=0.7, vox_mask=None, vox_lct=None,
				   obscenter=None, voxcenter=None, obswcs=None, voxwcs=None, seed=None, loud=True):

	[voxmin,nvox,dvox] = [loopid_info['voxmin'], loopid_info['nvox'], loopid_info['dvox']]
	nloops = len(loopid_info['looplengths'])
	if(obscenter is None): obscenter=map_in.center
	if(voxcenter is None): voxcenter=voxmap.center
	if(obswcs is None): obswcs=map_in.wcs
	if(voxwcs is None): voxwcs=voxmap.wcs
	rsun_cm = voxmap.rsun_meters.to(u.cm).value
	asec_cm = obscenter.observer.radius.to(u.cm).value*np.pi/180/3600.0 # (rsun_cm/(map_in.rsun_obs.to(u.radian).value))*np.pi/180.0/3600.0
	pxsz = 0.5*asec_cm*(map_in.scale[0]+map_in.scale[1]).to(u.arcsec/u.pixel).value
	loopflg = np.zeros(nloops+1,dtype='bool')+True
	loopflg[nloops]=False
	nl = loopid_info['lmax_scal']+1

	# --------------- Compute the detector matrix (PSF and pixelization):
	#print(map_in.data.shape, bin_fac, psf_size_px, pxsz)
	psfmat = get_psfmat(map_in.data.shape,bin_fac,psf_size_px,dvox,pxsz)
	outsize_hi = bin_fac*np.array(map_in.data.shape)
	nhi = outsize_hi[0]*outsize_hi[1]
	psfnorm = np.mean(np.sum(psfmat,axis=0).A1)
	psfnorm2 = np.mean(np.sum(psfmat,axis=1).A1)

	#---------------- Compute the Loop to pixel matrix (including PSF and pixelization):
	pxmin = vox2pix([0.0,0.0,0.0], voxmin, dvox, voxmap, map_in, obswcs=obswcs, voxcenter=voxcenter, vox_lct=vox_lct)
	pmax_x = vox2pix(np.array([nvox[0]-1.0,0.0,0.0]), voxmin, dvox, voxmap, map_in, obswcs=obswcs, voxcenter=voxcenter, vox_lct=vox_lct)
	pmax_y = vox2pix(np.array([0,nvox[1]-1.0,0.0]), voxmin, dvox, voxmap, map_in, obswcs=obswcs, voxcenter=voxcenter, vox_lct=vox_lct)
	pmax_z = vox2pix(np.array([0,0,nvox[2]-1.0]), voxmin, dvox, voxmap, map_in, obswcs=obswcs, voxcenter=voxcenter, vox_lct=vox_lct)
	xgrad = np.array([(pmax_x[0]-pxmin[0])/(nvox[0]-1.0),(pmax_y[0]-pxmin[0])/(nvox[1]-1.0),(pmax_z[0]-pxmin[0])/(nvox[2]-1.0)])
	ygrad = np.array([(pmax_x[1]-pxmin[1])/(nvox[0]-1.0),(pmax_y[1]-pxmin[1])/(nvox[1]-1.0),(pmax_z[1]-pxmin[1])/(nvox[2]-1.0)])

	[vox_ixa,vox_iya] = np.indices([nvox[0],nvox[1]],dtype='float64')
	vox_xy2 = ((vox_ixa-0.5*(nvox[0]-1))*dvox[0])**2 + ((vox_iya-0.5*(nvox[1]-1))*dvox[1])**2
	vox_iza = np.zeros(nvox[0:2],dtype='float64').flatten()
	[vox_ixa, vox_iya] = [vox_ixa.flatten(), vox_iya.flatten()]

	rgu = np.random.default_rng(seed)

	t0 = time.time()
	ncoef = nl*(nloops+1)
	loopids = loopid_info['voxel_loopids']
	seg_ids = loopid_info['voxel_loop_lengths']
	amat0 = csc_matrix((ncoef,nhi),dtype='float32')
	imin, imax = 0, loopids.shape[2]
	for i in range(imin, imax):
		loopid_slice = loopids[:,:,i].flatten()
		loopflg0 = loopflg[loopid_slice]
		segmentid_slice = seg_ids[:,:,i].flatten()
		loopid_flg = loopid_slice[loopflg0]
		segmentid_flg = segmentid_slice[loopflg0]
		vox_ixa2 = vox_ixa + rgu.uniform(low=-0.5,high=0.5,size=(vox_ixa.shape))
		vox_iya2 = vox_iya + rgu.uniform(low=-0.5,high=0.5,size=(vox_iya.shape))
		vox_iza2 = vox_iza + rgu.uniform(low=-0.5,high=0.5,size=(vox_iza.shape))
		ix_out = np.rint(bin_fac*(pxmin[0]+xgrad[0]*vox_ixa2+xgrad[1]*vox_iya2+xgrad[2]*(i-vox_iza2))).astype('int32')
		iy_out = np.rint(bin_fac*(pxmin[1]+ygrad[0]*vox_ixa2+ygrad[1]*vox_iya2+ygrad[2]*(i-vox_iza2))).astype('int32')
		inbounds = (ix_out >= 0)*(iy_out >= 0)*(ix_out < outsize_hi[0])*(iy_out < outsize_hi[1])
		if(not(vox_mask is None)): inbounds *= vox_mask[:,:,i].flatten()
		loopid_out = loopid_slice[inbounds].flatten().astype(np.int32)
		seg_id_out = segmentid_slice[inbounds].flatten().astype(np.int32)
		ixy = ix_out[inbounds]*outsize_hi[1]+iy_out[inbounds]
		inten_slice=loopflg0[inbounds]
		amat0 += csc_matrix((inten_slice,(loopid_out*nl+seg_id_out,ixy)),shape=(ncoef,nhi))
		amat0.sum_duplicates()            
		if(loud): print('Slice ',i,' of ',imax,time.time()-t0, 'Using: %s kb' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

	amat0 = (psfmat.transpose()*amat0.transpose()).transpose()

	modelinputs = {'map':map_in, 'vox_xy2':vox_xy2, 'psfnorm':psfnorm, 'psfnorm2':psfnorm2, 'rsun_cm':rsun_cm}	
	return amat0, modelinputs
	
