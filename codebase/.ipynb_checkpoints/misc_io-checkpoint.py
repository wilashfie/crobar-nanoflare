import copy, numpy as np, astropy.units as u; from coord_grid import coord_grid


# Massaging the x and y ranges:
def get_croprs(config, emis_pad=10):
	import numpy as np

	pos_unit = u.Unit(str(config['pos_unit']))
	[[x0, y0], [xsz0, ysz0]] = config['region_origin'], config['region_size']
	[xl, yl, xh, yh] = [x0, y0, x0+xsz0, y0+ysz0]
	boundpad = config['ivp_boundpad']
	cropr = [x0-xsz0*(boundpad[0,0]/(1-boundpad[0,0])),x0+xsz0/(1+boundpad[1,0]),
	        y0-ysz0*(boundpad[0,1]/(1-boundpad[0,1])),y0+ysz0/(1+boundpad[1,1])]*u.arcsec
	cropr0 = [x0, x0+xsz0, y0, y0+ysz0]*pos_unit
	# Crop the data for the reconstruction in a bit compared to the cube. This improves the ability of the 
	# reconstruction to fit the data, although it does also mean that the edges of the reconstruction are 
	# underconstrained. 
	cropr_emis = cropr0 + np.array([1,-1,1,-1])*emis_pad*u.arcsec	
	return cropr, cropr_emis

def get_map(path, cropr):
	#import sunpy.map.Map, astropy.coordinates.SkyCoord
	from astropy.coordinates import SkyCoord; from sunpy.map import Map
	map = Map(path)
	blc=SkyCoord(cropr[0],cropr[2],frame=map.coordinate_frame)
	trc=SkyCoord(cropr[1],cropr[3],frame=map.coordinate_frame)
	return map.submap(blc,top_right=trc)

def get_bounds(mag_coords, footpoint_config, loop_config):
	rsun_cm = footpoint_config['rsun_cm']

	# Set up boundaries for overall region:
	bounds0 = np.array([[np.min(c),np.max(c)] for c in mag_coords]).T
	bounds0[1,2] = bounds0[0,2] + footpoint_config['height_fac']*np.min(bounds0[1,0:2]-bounds0[0,0:2])
	
	heights = (mag_coords[0]**2+mag_coords[1]**2+(rsun_cm+mag_coords[2])**2)**0.5 - rsun_cm
	# Boundaries for field line tracing are set slightly larger to avoid
	# issues when an initial point falls outside the boundary:
	tracer_bounds = copy.deepcopy(bounds0)
	tracer_bounds[0,0] -= loop_config['tracer_pad']*(tracer_bounds[1,0]-tracer_bounds[0,0])
	tracer_bounds[0,1] -= loop_config['tracer_pad']*(tracer_bounds[1,1]-tracer_bounds[0,1])
	tracer_bounds[1,0] += loop_config['tracer_pad']*(tracer_bounds[1,0]-tracer_bounds[0,0])
	tracer_bounds[1,1] += loop_config['tracer_pad']*(tracer_bounds[1,1]-tracer_bounds[0,1])
	tracer_bounds[0,2] = np.max(heights)
	tracer_bounds[1,2] = tracer_bounds[0,2] + bounds0[1,2]-bounds0[0,2] # np.max(mag_coords[2,:,:,:]) - bounds0[0,2]
	return bounds0, tracer_bounds

# Get coordinate grids (see coord_grid.py for details of these)
def bounds2grid(bounds0,dvox,offsets=None,pad=None,frame=None):
	if(offsets is None): offsets = 0.5+0.0*dvox
	if(pad is None): pad = np.zeros([2,len(dvox)])
	if(frame is None): frame = np.arange(len(dvox))
	bounds = bounds0+pad*(bounds0[1]-bounds0[0])
	nvox = np.floor((bounds[1]-bounds[0])/dvox).astype(np.int32)
	vox_origin = bounds[0]+offsets*(bounds[1]-bounds[0]-nvox*dvox)
	return coord_grid(nvox,vox_origin,np.diag(dvox),frame)

