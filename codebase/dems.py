def simple_reg_dem(data, errors, exptimes, logt, tresps, guess_fac=1.0,
					kmax=100, kcon=5, steps0=[0.1,0.5], drv_con=8.0, chi2_th=1.0, tol=0.1):
	
	import numpy as np
	from scipy.linalg import cho_factor, cho_solve
	steps0 = np.array(steps0)

	[nt,nd] = tresps.shape
	nt_ones = np.ones(nt)
	[nx, ny, nd] = data.shape
	dT = logt[1:nt]-logt[0:nt-1]
	[dTleft, dTright] = [np.diag(np.hstack([dT,0])),np.diag(np.hstack([0,dT]))]
	[idTleft, idTright] = [np.diag(np.hstack([1.0/dT,0])),np.diag(np.hstack([0,1.0/dT]))]
	Bij = ((dTleft+dTright)*2.0 + np.roll(dTright,-1,axis=0) + np.roll(dTleft,1,axis=0))/6.0
	Rij = np.matmul((tresps*np.outer(nt_ones,exptimes)).T,Bij) # Matrix mapping coefficents to data
	Dij = idTleft+idTright - np.roll(idTright,-1,axis=0) - np.roll(idTleft,1,axis=0)
	regmat = Dij*nd/(drv_con**2*(logt[nt-1]-logt[0]))
	rvec = np.sum(Rij,axis=1)

	dems = np.zeros([nx,ny,nt])
	chi2 = np.zeros([nx,ny]) - 1.0
	for i in range(0,nx):
		print(i)
		for j in range(0,ny):
			err = errors[i,j,:]
			dat0 = np.clip(data[i,j,:],0.0,None)
			s = np.log(guess_fac*np.sum((rvec)*(np.clip(dat0, 1.0e-2,None)/err**2))/np.sum((rvec/err)**2)/nt_ones)
			for k in range(0,kmax):
				steps = steps0 # *(1-np.exp(-(k+1)/5))
				dat = (dat0-np.matmul(Rij,((1-s)*np.exp(s))))/err # Correct data by f(s)-s*f'(s)...
				mmat = Rij*np.outer(1.0/err,np.exp(s))
				amat = np.matmul(mmat.T,mmat)+regmat
				try: [c,low] = cho_factor(amat)
				except: break
				c2p = np.mean((dat0-np.dot(Rij,np.exp(s)))**2/err**2)
				deltas = cho_solve((c,low),np.dot(mmat.T,dat))-s
				deltas *= np.clip(np.max(np.abs(deltas)),None,0.5/steps[0])/np.max(np.abs(deltas))
				ds = 1-2*(c2p < chi2_th) # Direction sign; is chi squared too large or too small?
				c20 = np.mean((dat0-np.dot(Rij,np.exp(s+deltas*ds*steps[0])))**2.0/err**2.0)
				c21 = np.mean((dat0-np.dot(Rij,np.exp(s+deltas*ds*steps[1])))**2.0/err**2.0)
				interp_step = ((steps[0]*(c21-chi2_th)+steps[1]*(chi2_th-c20))/(c21-c20))
				s += deltas*ds*np.clip(interp_step,steps[0],steps[1])
				chi2[i,j] = np.mean((dat0-np.dot(Rij,np.exp(s)))**2/err**2)
				if((ds*(c2p-c20)/steps[0] < tol)*(k > kcon) or np.abs(chi2[i,j]-chi2_th) < tol): break
			dems[i,j,:] = np.exp(s)

	return dems,chi2

def estimate_uncertainty(errs, logt, tresps, dlogt=0.1, project_map=None):

	import numpy as np

	nt = len(logt)
	nx, ny, nchan = errs.shape
	output_cube = np.zeros([nx,ny,nt])

	for i in range(0,nchan):
		tresp = tresps[:,i]
		for j in range(0,len(logt)):
			output_cube[:,:,j] += (tresp[j]*dlogt/errs[:,:,i])**2

	return np.sqrt(nchan/output_cube)
