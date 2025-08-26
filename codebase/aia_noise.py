import numpy as np, astropy.units as u


def aia_noise(image, channel=None):
    
    if(channel is None): #channel = map_in.meta['detector']+map_in.meta['wave_str']
         print("Warning: No channel specified, using default AIA 171 channel.")
         channel = 'AIA171_THIN'
    refchannels=np.array(['AIA94_THIN', 'AIA131_THIN', 'AIA171_THIN', 'AIA193_THIN', 'AIA211_THIN', 'AIA304_THIN', 'AIA335_THIN'])
    refg = np.array([2.128,1.523,1.168,1.024,0.946,0.658,0.596])
    refn = np.array([1.14,1.18,1.15,1.2,1.2,1.14,1.18])

    dnpp = refg[np.where(refchannels == channel)]
    rdn = refn[np.where(refchannels == channel)]

    photon_counts = np.clip(image / dnpp, 0.0, None)
    noisy_photons = np.random.poisson(lam=photon_counts)
    poisson_data = noisy_photons * dnpp

    read_noise = np.random.normal(0, rdn, size=image.shape)

    noisy_data = poisson_data + read_noise

    #noise_floor = 0.1  # Small positive value
    #noisy_data_floored = np.where(noisy_data <= 0, noise_floor, noisy_data)
    
    return noisy_data
