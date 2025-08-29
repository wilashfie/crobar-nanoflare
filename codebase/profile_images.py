import os, copy, sunpy, numpy as np, matplotlib.pyplot as plt
from datetime import datetime, timedelta
from aia_noise import aia_noise


def image_gen(i, amat_gen, evaluator, trlogt, tresps, basemaps, channels, dvox, asec_cm, nl=255):
	nloops = evaluator.nloop
	temps, rhos = np.zeros([nloops+1,nl]), np.zeros([nloops+1,nl])
	for l in range(0,nloops): temps[l,:], rhos[l,:], pres, g = evaluator.calculate(l,i)

	images=[]
	for j in range(0,len(channels)):
		profs = np.interp(np.log10(temps),trlogt,tresps[j])*rhos**2
		dimension_factor = np.prod(dvox[0:2])/(asec_cm*asec_cm*basemaps[j].meta['CDELT1']*basemaps[j].meta['CDELT2'])
		image = basemaps[j].meta['exptime']*amat_gen.T.dot(profs.flatten()).reshape(basemaps[j].data.shape)*dvox[2]*dimension_factor
		image = aia_noise(image, channel=channels[j])
		images.append(image)

	return(images)

def write_files(t, images, basemaps, channels, base_time, fits_dir='ebtel_fits/', image_dir='images/', gfac=1.0/2.2):
	new_time = base_time + timedelta(seconds=int(t))
	date_obs_string = new_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]	
	for j in range(0,len(channels)):
		save_string = 'ebtel_'+channels[j]+'_'+new_time.strftime('%Y%m%dT%H%M%S')+'.fits'
		meta = copy.deepcopy(basemaps[j].meta)
		meta['date-obs'] = date_obs_string

		outmap = sunpy.map.Map(images[j],meta)
		outmap.save(os.path.join(fits_dir,save_string),overwrite=True)

		image_dir = 'images/'
		fig = plt.figure(figsize=(8,8))		
		plt.imshow(np.clip(outmap.data,0,None)**gfac,cmap=plt.get_cmap('gray'), vmin=0, vmax=3000**gfac)
		plt.title(channels[j]+' '+date_obs_string)	
		fig.savefig(image_dir + 'ebtel_'+channels[j]+'_'+new_time.strftime('%Y%m%dT%H%M%S')+'.png')
		plt.close()
