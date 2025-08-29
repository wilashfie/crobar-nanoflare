import os, copy, sunpy, numpy as np, matplotlib.pyplot as plt
from datetime import datetime, timedelta
from aia_noise import aia_noise
import astropy.units as u


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

		#save_string = 'ebtel_'+channels[j]+'_'+new_time.strftime('%Y%m%dT%H%M%S')+'.fits'
		save_string = 'ebtel_'+channels[j].split('_')[0]+'_'+new_time.strftime('%Y%m%dT%H%M%S')+'.fits'

		meta = copy.deepcopy(basemaps[j].meta)
		meta['date-obs'] = date_obs_string
		meta['wavelnth'] = meta['wavelnth']
		meta['wave_str'] = channels[j] # f'{meta['wavelnth']}_THIN'

		temp_map = sunpy.map.Map(images[j], meta)
		center_x, center_y = temp_map.data.shape[1] // 2, temp_map.data.shape[0] // 2
		
		x_start = center_x - 256
		x_end = center_x + 255  
		y_start = center_y - 256
		y_end = center_y + 255
		
		bl = temp_map.pixel_to_world(x_start*u.pixel,y_start*u.pixel)
		tr = temp_map.pixel_to_world(x_end*u.pixel,y_end*u.pixel)
		
		outmap = temp_map.submap(bl, top_right=tr)
		outmap.save(f'{fits_dir}/{channels[j].split('_')[0]}/{save_string}', overwrite=True)

		fig = plt.figure(figsize=(8,8))		
		plt.imshow(np.clip(outmap.data,0,None)**gfac,cmap=plt.get_cmap('gray'), vmin=0, vmax=3000**gfac)
		plt.title(channels[j]+' '+date_obs_string)	
		img_string = 'ebtel_'+channels[j].split('_')[0]+'_'+new_time.strftime('%Y%m%dT%H%M%S')+'.png'
		fig.savefig(f'{image_dir}/{channels[j].split('_')[0]}/{img_string}',dpi=150)
		plt.close()
