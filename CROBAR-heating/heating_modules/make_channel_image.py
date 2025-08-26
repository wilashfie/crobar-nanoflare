import numpy as np
from sunpy.coordinates import HeliographicStonyhurst, Heliocentric
from processify import processify
import astropy.units as u

def heeq_to_hcc(x_heeq, y_heeq, z_heeq, observer_coordinate):
    """
    Convert Heliocentric Earth Equatorial (HEEQ) coordinates to Heliocentric
    Cartesian Coordinates (HCC) for a given observer. See Eqs. 2 and 11 of [1]_.

    References
    ----------
    .. [1] Thompson, W. T., 2006, A&A, `449, 791 <http://adsabs.harvard.edu/abs/2006A%26A...449..791T>`_
    """
    observer_coordinate = observer_coordinate.transform_to(HeliographicStonyhurst)
    Phi_0 = observer_coordinate.lon.to(u.radian)
    B_0 = observer_coordinate.lat.to(u.radian)

    x_hcc = y_heeq*np.cos(Phi_0) - x_heeq*np.sin(Phi_0)
    y_hcc = z_heeq*np.cos(B_0) - x_heeq*np.sin(B_0)*np.cos(Phi_0) - y_heeq*np.sin(Phi_0)*np.sin(B_0)
    z_hcc = z_heeq*np.sin(B_0) + x_heeq*np.cos(B_0)*np.cos(Phi_0) + y_heeq*np.cos(B_0)*np.sin(Phi_0)

    return x_hcc, y_hcc, z_hcc


@u.quantity_input
def heeq_to_hcc_coord(x_heeq: u.cm, y_heeq: u.cm, z_heeq: u.cm, observer_coordinate):
    """
    Return an HCC `~astropy.coordinates.SkyCoord` object from a set of HEEQ positions.
    This is a wrapper around `~heeq_to_hcc`.
    """
    x, y, z = heeq_to_hcc(x_heeq, y_heeq, z_heeq, observer_coordinate)
    return SkyCoord(x=x, y=y, z=z, frame=Heliocentric(observer=observer_coordinate))



def vox2pix(vox_in,voxmin,dvox,map_in,**kwargs):
	
	lon = kwargs.get('lon',map_in.center.heliographic_stonyhurst.lon)
	lat = kwargs.get('lat',map_in.center.heliographic_stonyhurst.lat)
	t0 = kwargs.get('obstime',map_in.center.heliographic_stonyhurst.obstime)
	rep = kwargs.get('representation','cartesian')

	vox = np.expand_dims((vox_in*dvox+voxmin),1)*u.cm

	cen_new = SkyCoord(sunpy.coordinates.frames.HeliographicStonyhurst(lon=lon, lat=lat, obstime=t0), representation=rep)
	cheeq = from_local(vox[0],vox[1],vox[2],cen_new).cartesian.xyz.T
	chobs=(heeq_to_hcc_coord(cheeq[:,0],cheeq[:,1],cheeq[:,2],map_in.observer_coordinate)).transform_to(map_in.coordinate_frame)        

	pxobj = map_in.world_to_pixel(chobs)

	return np.array([pxobj.y.value[0],pxobj.x.value[0]])
	
@processify
def make_channel_img(t0):

	tstart=time.time()
	map_in=mapaia_crop0
	voxmin = loopid_info['voxmin']
	nvox = loopid_info['nvox']
	dvox = loopid_info['dvox']
	voxpx_ratio = (aia_pxsz**2/dvox[0]/dvox[1])**0.5
	bin_fac = 4

	from skimage.transform import resize 
	map_template=copy.deepcopy(map_in)
	outsize = np.array(map_template.data.shape)
	nvox = loopid_info['nvox']
	inten_img = np.zeros(nvox[0:2],dtype='float32')
	aiapsf_size_px = aiapsf_size/0.6#*voxpx_ratio

	pxmin = vox2pix([0,0,0],voxmin,dvox,map_template)
	pmax_x = vox2pix(np.array([nvox[0]-1,0,0]),voxmin,dvox,map_template)
	pmax_y = vox2pix(np.array([0,nvox[1]-1,0]),voxmin,dvox,map_template)
	pmax_z = vox2pix(np.array([0,0,nvox[2]-1]),voxmin,dvox,map_template)

	xgrad = np.array([(pmax_x[0]-pxmin[0])/(nvox[0]-1),(pmax_y[0]-pxmin[0])/(nvox[1]-1),(pmax_z[0]-pxmin[0])/(nvox[2]-1)])
	ygrad = np.array([(pmax_x[1]-pxmin[1])/(nvox[0]-1),(pmax_y[1]-pxmin[1])/(nvox[1]-1),(pmax_z[1]-pxmin[1])/(nvox[2]-1)])

	inten_rescale = bin_fac**2
	outsize_hi = bin_fac*outsize
	nhi = outsize_hi[0]*outsize_hi[1]
	outimg_hi = np.zeros(nhi)

	[vox_ixa,vox_iya] = np.indices([nvox[0],nvox[1]])
	vox_iza = np.zeros(nvox[0]*nvox[1],dtype='int32')
	vox_ixa = vox_ixa.flatten()
	vox_iya = vox_iya.flatten()

	looptemps = np.zeros([nloops+1,nlbins2])
	loopdens = np.zeros([nloops+1,nlbins2])
	for index in range(0,len(ar.loops)):
		timeindex = np.argmin(np.abs(ar.loops[index].time-t0))
		profile = compute_loop_profiles(index,ar.loops,ebtelterms,segment_heights,segment_relareas,lbin_cents,time_indices=timeindex)
		looptemps[index,:] = profile['temp'][0,:]
		loopdens[index,:] = profile['dens'][0,:]
		
	for i in range(0,nvox[2]):
		inten_slice = np.zeros([nvox[0],nvox[1]])
		loopid_slice = loopids[:,:,i]
		segmentid_slice = voxlengths[:,:,i]
		loopflg0 = loopflg[loopid_slice]
		loopid_flg = loopid_slice[loopflg0]
		segmentid_flg = segmentid_slice[loopflg0]
		densimg = loopdens[loopid_flg,segmentid_flg]**2
		tempimg = looptemps[loopid_flg,segmentid_flg]
		inten_slice[loopflg0] = np.interp(tempimg,trtemp,tresp)*densimg
		ix_out = np.round(bin_fac*(pxmin[0]+xgrad[0]*vox_ixa+xgrad[1]*vox_iya+xgrad[2]*(vox_iza+i))).astype('int32')
		iy_out = np.round(bin_fac*(pxmin[1]+ygrad[0]*vox_ixa+ygrad[1]*vox_iya+ygrad[2]*(vox_iza+i))).astype('int32')
		outimg_hi += np.bincount(ix_out*outsize_hi[1]+iy_out,weights=inten_slice.flatten(),minlength=nhi)

	outimg_hi=np.reshape(outimg_hi,outsize_hi)

	implt = ndimage.filters.gaussian_filter(np.abs(outimg_hi),aiapsf_size_px*bin_fac)

	[ix_hi,iy_hi] = np.indices(outsize_hi)
	ix = np.floor(ix_hi.flatten()/bin_fac).astype('int32')
	iy = np.floor(iy_hi.flatten()/bin_fac).astype('int32')
	outimg = np.bincount(ix*outsize[1]+iy,weights=np.reshape(implt,nhi),minlength=outsize[0]*outsize[1])
	return np.reshape(outimg,outsize)*dvox[2]/voxpx_ratio**2
