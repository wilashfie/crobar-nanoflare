from sunpy.coordinates import HeliographicStonyhurst, Heliocentric


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

def get_kernel_2d(size0,dvox_in):
	dvox = dvox_in[0:2]
	ev_rad = np.array(0.5*size0/dvox)
	kern_rtot = 4*np.ceil(ev_rad).astype('int32')
	sz_voxels = 2*kern_rtot+1
	area_voxels = np.prod(sz_voxels)
	[vox_ixa,vox_iya] = np.indices(sz_voxels,dtype='int32')
	vox_ixa0 = (vox_ixa.flatten())-kern_rtot[0]
	vox_iya0 = (vox_iya.flatten())-kern_rtot[1]
	event_nvox = len(vox_ixa0)
	kernel = np.exp(-0.5*(((vox_ixa0)/(ev_rad[0]))**2+((vox_iya0)/(ev_rad[1]))**2))
	kernel = kernel/np.sum(kernel)
	kflat = kernel.flatten()
	wkern = np.where(kernel > 5.0e-3/event_nvox)
	kflat = kflat[wkern].astype('float32')
	vox_ixa0 = vox_ixa0[wkern]
	vox_iya0 = vox_iya0[wkern]
	vox_ioffs = np.vstack([vox_ixa0,vox_iya0]).astype('int32')
	event_nvox = len(vox_ixa0)
	kernones = 1+np.zeros(event_nvox,dtype='int32')
	return vox_ioffs,kflat,ev_rad,area_voxels,kernones,np.reshape(kernel,sz_voxels)

@processify
def get_heating_distribution(heatmap_name_in,loopids,loopid_info,Bfield_yt,heating_distribution_name,recompute_heating_distribution=False,return_projection=True):
	
	if(recompute_heating_distribution or os.path.exists(heating_distribution_name) == 0):
		[hmap,hmxyz,hmimg] = compute_heating_distribution(heatmap_name_in,loopids,loopid_info,Bfield_yt)
		pickle.dump([heatmap_name_in,hmap,hmxyz,hmimg],open(heating_distribution_name,"wb"))
		heatmap_name = heatmap_name_in
	else:
		[heatmap_name,hmap,hmxyz,hmimg] = pickle.load(open(heating_distribution_name,"rb"))
		if(heatmap_name_in != heatmap_name): print('Warning: input heatmap name, ',heatmap_name_in,', does not match name, ',heatmap_name,', in stored heating distribution file, ',heating_distribution_name)
	if(return_projection): # Only return a projection to reduce memory usage (i.e., initial call to compute or show heating)
		return np.flipud(np.sum(np.abs(hmap),axis=2).transpose()), hmimg, heatmap_name
	else:
		return hmap,hmxyz,hmimg

def load_heating_distribution(heating_distribution_name):
	[heatmap_name,hmap,hmxyz,hmimg] = pickle.load(open(heating_distribution_name,"rb"))
	return [heatmap_name,hmap,hmxyz,hmimg]
	
def compute_heating_distribution(heatmap_name,loopids,loopid_info,Bfield_yt):

	from scipy.sparse import diags
	from scipy.sparse.linalg import bicgstab
	from scipy.sparse import csr_matrix
    
	[lbin_cents,segment_relareas,segment_heights,loop_amaxes,area_norms] = find_loop_areas(loopids,loopid_info,voxlengths)
	voxmin = loopid_info['voxmin']
	nvox = loopid_info['nvox']
	dvox = loopid_info['dvox']
	bin_fac=1
	map_in = get_cropmap0(base_path+heatmap_name,cropr)
	#map_in.data[::] = map_in.data**2    
	thold = np.sort(map_in.data.flatten())[round(5.0e-3*len(map_in.data.flatten()))]
	map_in.data[::] = np.abs(map_in.data-map_in.data*np.arctan(thold/(0.1+np.abs(map_in.data))))
	#map_in.data[::] = ndimage.filters.gaussian_filter(map_in.data,3)
	weights = ((25*map_in.data)**2+0.25*np.median(map_in.data)**4)**0.5
	outsize = np.array(map_in.data.shape)
	weights_flat = weights.flatten()
	loopflg = np.zeros(nloops+1,dtype='bool')+True
	loopflg[nloops]=False
	loopflg[np.where(loopid_info['looplengths'] < 3*np.max(dvox))]=False

	# --------------- Compute the detector matrix (PSF and pixelization):

	outsize_hi = (bin_fac*outsize).astype('int64')
	nhi = outsize_hi[0]*outsize_hi[1]
	outimg_hi = np.zeros(nhi)
	aiapsf_size_px = aiapsf_size/0.6
	[xa0,ya0] = np.indices(outsize_hi)
	xa0 = np.array(xa0).flatten()
	ya0 = np.array(ya0).flatten()

	nlo = outsize[0]*outsize[1]
	[vox_ioffs,kflat,ev_rad,area_voxels,kernones,kernel] = get_kernel_2d((aiapsf_size_px**2+((5/np.pi)*dvox[0]/aia_pxsz)**2)**0.5,np.array([1,1])/bin_fac)

	kernsize = len(kflat)
	indices = np.zeros([nhi,kernsize],dtype='uint32')
	kernvals = np.zeros([nhi,kernsize],dtype='float32')
	for i in range(0,kernsize):
		xa2 = np.rint((xa0+vox_ioffs[0][i])/bin_fac)
		ya2 = np.rint((ya0+vox_ioffs[1][i])/bin_fac)
		boundflg = (xa2>=0)*(ya2>=0)*(xa2<outsize[0])*(ya2<outsize[1])
		indices[:,i] = boundflg*(xa2*outsize[1]+ya2)
		kernvals[:,i] = kflat[i]
				
	#---------------- Compute the Loop to pixel matrix (including PSF and pixelization):

	voxpx_ratio = (aia_pxsz**2/dvox[0]/dvox[1])**0.5
	hfac = 60e8/1.0e6

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

	outsize_hi = bin_fac*outsize
	nhi = outsize_hi[0]*outsize_hi[1]
	outimg_hi = np.zeros(nhi)

	[vox_ixa,vox_iya] = np.indices([nvox[0],nvox[1]])
	vox_iza = np.zeros(nvox[0]*nvox[1],dtype='int32')
	vox_ixa = vox_ixa.flatten()
	vox_iya = vox_iya.flatten()

	#amat0 = csr_matrix((nloops+1,nlo),dtype='float32')
	amat0 = csr_matrix((2*nloops+1,nlo),dtype='float32')
	temp1 = 1.5e6
	temp0 = 0.75e6
	xpo1 = 2.1
	xpo = 1.4
	
	tprof = (((4*(lbin_cents)*(1-lbin_cents))+0.05)/1.05)**1.0
	kernones = np.zeros((1,kernsize),dtype='int32')+1
	t1=0
	t2=0
	t3=0
	for i in range(0,nvox[2]):
		tlast=time.time()
		pscal = np.exp(-i*dvox[2]/(temp0*hfac))
		pscal2 = np.exp(-i*dvox[2]/(temp1*hfac))
		inten_slice = np.zeros([nvox[0],nvox[1]],dtype='float32')
		inten_slice2 = np.zeros([nvox[0],nvox[1]],dtype='float32')
		loopid_slice = loopids[:,:,i]
		segmentid_slice = voxlengths[:,:,i]
		loopflg0 = loopflg[loopid_slice]
		loopid_flg = loopid_slice[loopflg0]
		segmentid_flg = segmentid_slice[loopflg0]
		t1+=time.time()-tlast
		tlast = time.time()
		##densimg = tprof[segmentid_flg]**xpo*(pscal/tprof[segmentid_flg])**2 # loopdens[loopid_flg,segmentid_flg]**2
		#densimg = (temp0*tprof[segmentid_flg])**xpo/(temp0**xpo+(temp0*tprof[segmentid_flg])**xpo)*(pscal/tprof[segmentid_flg])**2 # loopdens[loopid_flg,segmentid_flg]**2
		#densimg2 = (temp1*tprof[segmentid_flg])**xpo1/(temp1**xpo1+(temp1*tprof[segmentid_flg])**xpo1)*(pscal2/tprof[segmentid_flg])**2 # loopdens[loopid_flg,segmentid_flg]**2
		densimg = (tprof[segmentid_flg])**xpo*(pscal/tprof[segmentid_flg])**2 # loopdens[loopid_flg,segmentid_flg]**2
		densimg2 = (tprof[segmentid_flg])**xpo1*(pscal2/tprof[segmentid_flg])**2 # loopdens[loopid_flg,segmentid_flg]**2
		##tempimg = looptemps[loopid_flg,segmentid_flg]
		inten_slice[loopflg0] = densimg
		inten_slice2[loopflg0] = densimg2
		ix_out = np.rint(bin_fac*(pxmin[0]+xgrad[0]*vox_ixa+xgrad[1]*vox_iya+xgrad[2]*(vox_iza+i))).astype('int32')
		iy_out = np.rint(bin_fac*(pxmin[1]+ygrad[0]*vox_ixa+ygrad[1]*vox_iya+ygrad[2]*(vox_iza+i))).astype('int32')
		inbounds = (ix_out >= 0)*(iy_out >= 0)*(ix_out < outsize_hi[0])*(iy_out < outsize_hi[1])
		inds_out = indices[inbounds*ix_out*outsize_hi[1]+inbounds*iy_out,:].flatten()
		kvals_out = ((np.expand_dims((inbounds*inten_slice.flatten()),axis=1)*kernones)*kernvals[ix_out*outsize_hi[1]+iy_out,:]).flatten()
		kvals_out2 = ((np.expand_dims((inbounds*inten_slice2.flatten()),axis=1)*kernones)*kernvals[ix_out*outsize_hi[1]+iy_out,:]).flatten()
		loopid_out = (np.expand_dims(loopid_slice.flatten(),axis=1)*kernones).flatten()
		t2+=time.time()-tlast
		tlast = time.time()
		#a=csr_matrix((kvals_out,(loopid_out,inds_out)),shape=(nloops+1,nlo))
		#a.sum_duplicates()
		#amat0 += a
		a=csr_matrix((kvals_out,(2*loopid_out,inds_out)),shape=(2*nloops+1,nlo))
		a.sum_duplicates()
		amat0 += a
		a=csr_matrix((kvals_out2,(2*loopid_out+1,inds_out)),shape=(2*nloops+1,nlo))
		a.sum_duplicates()
		amat0 += a
		t3+=time.time()-tlast
		print('Slice ',i,' of ',nvox[2],t1,t2,t3)

	# --------------------- Matrix formed, compute initial guess:
	loops = bicgstab(amat0*diags(1.0/weights_flat**2)*amat0.transpose(),amat0.dot((map_in.data.flatten())**2/weights_flat**2),tol=0.05)
	guess = np.abs(loops[0])+0.01

	# --------------------- Now do the iteration:
	for i in range(0,40):
		rmat = diags(guess)*amat0*diags(1.0/weights_flat)
		bvec = rmat.dot((map_in.data.flatten()**2-amat0.transpose().dot(guess-np.log(guess)*guess))/weights_flat)

		loops = bicgstab(rmat*rmat.transpose(),bvec,np.log(guess),tol=1.0e-4*(1+(1.0e-2/1.0e-4)/(i+1)))
		guess = np.exp(np.log(guess)+0.25*(loops[0]-np.log(guess)))
		print(i,np.mean(loops[0]),np.std(loops[0]),np.min(loops[0]),np.max(loops[0]),np.min(tprof)*1e6)   
	
	hmx = Bfield_yt.domain_left_edge[0]+(0.5+np.arange(nvox[0]))*dvox[0]
	hmy = Bfield_yt.domain_left_edge[1]+(0.5+np.arange(nvox[1]))*dvox[1]
	hmz = Bfield_yt.domain_left_edge[2]+(0.5+np.arange(nvox[2]))*dvox[2]
	heatmap = np.zeros(nvox,dtype='float32')
	for i in range(0,nvox[2]):
		pscal = np.exp(-i*dvox[2]/(temp0*hfac))
		pscal2 = np.exp(-i*dvox[2]/(temp1*hfac))
		inten_slice = np.zeros([nvox[0],nvox[1]],dtype='float32')
		inten_slice2 = np.zeros([nvox[0],nvox[1]],dtype='float32')
		loopid_slice = loopids[:,:,i]
		segmentid_slice = voxlengths[:,:,i]
		loopflg0 = loopflg[loopid_slice]
		loopid_flg = loopid_slice[loopflg0]
		segmentid_flg = segmentid_slice[loopflg0]
		##inten_slice[loopflg0] = guess[loopid_flg]*tprof[segmentid_flg]**xpo*(pscal/tprof[segmentid_flg])**2 # loopdens[loopid_flg,segmentid_flg]**2
		#inten_slice[loopflg0] = guess[2*loopid_flg]*(temp0*tprof[segmentid_flg])**xpo/(temp0**xpo+(temp0*tprof[segmentid_flg])**xpo)*(pscal/tprof[segmentid_flg])**2 # loopdens[loopid_flg,segmentid_flg]**2
		#inten_slice2[loopflg0] = guess[2*loopid_flg+1]*(temp1*tprof[segmentid_flg])**xpo1/(temp1**xpo1+(temp1*tprof[segmentid_flg])**xpo1)*(pscal2/tprof[segmentid_flg])**2 # loopdens[loopid_flg,segmentid_flg]**2

		#inten_slice[loopflg0] = guess[loopid_flg]*(temp0*tprof[segmentid_flg])**xpo*(pscal/tprof[segmentid_flg])**2 # loopdens[loopid_flg,segmentid_flg]**2
		#heatmap[:,:,i] += inten_slice
		inten_slice[loopflg0] = guess[2*loopid_flg]*(tprof[segmentid_flg])**xpo*(pscal/tprof[segmentid_flg])**2 # loopdens[loopid_flg,segmentid_flg]**2
		inten_slice2[loopflg0] = guess[2*loopid_flg+1]*(tprof[segmentid_flg])**xpo1*(pscal2/tprof[segmentid_flg])**2 # loopdens[loopid_flg,segmentid_flg]**2
		heatmap[:,:,i] += inten_slice
		heatmap[:,:,i] += inten_slice2
	
	return (heatmap/dvox[2]).astype('float32'),(hmx.astype('float32'),hmy.astype('float32'),hmz.astype('float32')),map_in.data**2
