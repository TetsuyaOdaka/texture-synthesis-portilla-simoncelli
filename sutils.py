import numpy as np
import math
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA 
import os
import logging

SCRIPT_NAME = os.path.basename(__file__)

# logging
LOG_FMT = "[%(name)s] %(asctime)s %(levelname)s %(lineno)s %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FMT)
LOGGER = logging.getLogger(os.path.basename(__file__))

PS = 1e-6

'''
	Upsampling

	this is a port of textureSynth/expand.m by J. Portilla and E. Simoncelli.
	http://www.cns.nyu.edu/~lcv/texture/
	https://github.com/LabForComputationalVision/textureSynth

'''
def expand(t ,f, p=0):
	my,mx = t.shape
	T = np.zeros((my, mx), dtype=complex)
	my = f*my
	mx = f*mx
	Te = np.zeros((my, mx), dtype=complex)

	T = f**2 * np.fft.fftshift(np.fft.fft2(t))
	y1 = my/2 + 2 - my/(2*f)
	y2 = my/2 + my/(2*f)
	x1 = mx/2 + 2 - mx/(2*f)
	x2 = mx/2 + mx/(2*f)
	y1 = int(y1)
	y2 = int(y2)
	x1 = int(x1)
	x2 = int(x2)
    
	Te[y1-1:y2, x1-1:x2] = T[1:int(my/f), 1:int(mx/f)]
	Te[y1-2, x1-1:x2] = T[0, 1:int(mx/f)]/2
	Te[y2, x1-1:x2] = T[0, int(mx/f):0:-1]/2
	Te[y1-1:y2, x1-2] = T[1: int(my/f), 0]/2
	Te[y1-1:y2, x2] = T[int(my/f):0:-1, 0]/2

	esq = T[0,0] / 4
	Te[y1-2, x1-2] = esq
	Te[y1-2, x2] = esq
	Te[y2, x1-2] = esq
	Te[y2, x2] = esq

	Te = np.fft.fftshift(Te)
	te = np.fft.ifft2(Te)
	te = te.real

	return te


'''
	Downsampling

	this is a port of textureSynth/shrink.m by J. Portilla and E. Simoncelli.
	https://github.com/LabForComputationalVision/textureSynth

	http://www.cns.nyu.edu/~lcv/texture/

'''
def shrink(t, f):
	my,mx = t.shape
	T=np.fft.fftshift(np.fft.fft2(t))/f**2
	Ts=np.zeros((int(my/f), int(mx/f)), dtype=complex)
	y1=int(my/2 + 2 - my/(2*f))
	y2=int(my/2 + my/(2*f))
	x1=int(mx/2 + 2 - mx/(2*f))
	x2=int(mx/2 + mx/(2*f))

	Ts[1:int(my/f), 1:int(mx/f)] = T[y1-1:y2 ,x1-1:x2]
	Ts[0,1:int(mx/f)]=(T[y1-2, x1-1:x2]+T[y2, x1-1:x2])/2
	Ts[1:int(my/f),0] = (T[y1-1:y2, x1-2] + T[y1-1:y2, x2])/2
	Ts[0,0] = (T[y1-2,x1-1] + T[y1-2,x2] + T[y2, x1-1] + T[y2, x2+1])/4
	Ts=np.fft.fftshift(Ts)
	ts=np.fft.ifft2(Ts)
	ts = ts.real
#	ts = np.abs(ts)

	return ts

'''
	Doubling phases

	this is a port of textureSynth/modskew.m by J. Portilla and E. Simoncelli.
	https://github.com/LabForComputationalVision/textureSynth

'''
def double_phase(image):
	
	_rtmp = image.real
	_itmp = image.imag

	_theta = np.arctan2(_itmp, _rtmp)
	_rad = np.sqrt(_rtmp**2 + _itmp**2)

	_tmp = _rad * np.exp(2 * complex(0,1) * _theta)

	return _tmp

#def double_phase_ng(image):
#	ft = np.fft.fft2(image)
#	_ft = np.fft.fftshift(ft)	
#	
#	tmp_theta = np.angle(_ft)
#	tmp_pol = np.absolute(_ft)
#	_tmp = tmp_pol * np.exp(2 * complex(0,1) * tmp_theta)
#
#	_tmp = np.fft.ifft2(np.fft.ifftshift(_tmp))
#
#	return _tmp


'''
	modify skewness

	this is a port of textureSynth/modskew.m by J. Portilla and E. Simoncelli.
	https://github.com/LabForComputationalVision/textureSynth

	http://www.cns.nyu.edu/~lcv/texture/

'''
def mod_skew(im, sk):
	# mu
	_mean = np.mean(im.flatten())

	im = im - _mean

	_tmp = im**2
	_sd = np.sqrt(np.mean(_tmp.flatten()))

	mu = [im**i for i in range(3,7)]
	mu = [np.mean(mu[i].flatten()) for i in range(len(mu))]

#	print(im)
#	print(mu[0])
#	print(_sd)
	if _sd > 0:
		_sk = mu[0] / (_sd)**3

		_A = mu[3] - 3.*_sd*_sk*mu[2] + 3.*_sd**2.*(_sk**2.-1.)*mu[1] + _sd**6*(2 + 3*_sk**2 - _sk**4)
		_B = 3.*( mu[2] - 2.*_sd*_sk*mu[1] + _sd**5*_sk**3 )
		_C = 3.*( mu[1] - _sd**4*( 1. + _sk**2) )
		_D = _sk * _sd**3

		a = np.zeros_like(range(0,7), dtype='double')
		a[6] = _A**2.
		a[5] = 2.*_A*_B
		a[4] = _B**2 + 2.*_A*_C
		a[3] = 2.*(_A*_D + _B*_C)
		a[2] = _C**2 + 2.*_B*_D
		a[1] = 2.*_C*_D
		a[0] = _D**2

		_a2 = _sd**2
		_b2 = mu[1] - (1. + _sk**2)*_sd**4

		b = np.zeros_like(range(0,7), dtype='double')
		b[6] = _b2**3
		b[4] = 3.*_a2*_b2**2
		b[2] = 3.*_a2**2*_b2
		b[0] = _a2**3

		d = np.zeros_like(range(0,8), dtype='double')
		d[7] = _B * b[6]
		d[6] = 2*_C*b[6] - _A*b[4]
		d[5] = 3*_D*b[6]
		d[4] = _C*b[4] - 2.*_A*b[2]
		d[3] = 2*_D*b[4] - _B*b[2]
		d[2] = -3.*_A*b[0]
		d[1] = _D*b[2] - 2*_B*b[0]
		d[0] = -_C*b[0]

		d = d[::-1]
		mMlambda = np.roots(d)

		tg = mMlambda.imag / mMlambda.real
		_idx = np.where(np.abs(tg) < 1e-6)
		mMlambda = mMlambda[_idx].real
		lNeg = mMlambda[np.where(mMlambda < 0)]
		if lNeg.shape[0] == 0:
			lNeg = -1/2**-50

		lPos = mMlambda[np.where(mMlambda >= 0)]
		if lPos.shape[0] == 0:
			lPos = 1/2**-50

		lmi = np.max(lNeg)
		lma = np.min(lPos)

		lam = np.array([lmi, lma], dtype='double')

		mMnewSt = np.polyval(np.array([_A, _B, _C, _D], dtype='double'), lam) / np.sqrt(np.polyval(b[::-1], lam))

		skmin = np.min(mMnewSt)
		skmax = np.max(mMnewSt)

# Given a desired skewness, solves for lambda
		if sk <= skmin:
			lam = lmi
			LOGGER.debug('Saturating (down) skewness!')
		elif sk >= skmax:
			lam = lma
			LOGGER.debug('Saturating (up) skewness!')
		else:
			c = a - b*sk**2
			c = c[::-1]

			r = np.roots(c)

# Chose the real solution with minimum absolute value with the rigth sign
			lam = np.array( [0.] )
			co = 0
			tg = np.abs(r.imag / r.real)
			_idx = np.where(( np.abs(tg) < 1e-6 ) & ( np.sign(r.real) == np.sign(sk - _sk)))
			if r[_idx].shape[0] > 0:
				lam = r[_idx].real

			if np.all(lam == 0.):
				LOGGER.info('Warning: Skew adjustment skipped!')

			p = [_A, _B, _C, _D]

			if lam.shape[0] > 1:
				foo = np.sign(np.polyval(p, lam))
				if np.any(foo == 0):
					lam = lam[np.where(foo == 0)]
				else:
					lam = lam[np.where(foo == np.sign(sk))]		# rejects the symmetric solution

				if lam.shape[0] > 0:
					lam = lam[np.where(np.abs(lam) == np.min(abs(lam)))]	# the smallest that fix the skew
					lam = lam[0]
				else:
					lam = 0.

# Modify the channel
			chm = im + lam*(im**2 - _sd**2 - _sd*_sk*im)		# adjust the skewness
			chm = chm * _sd / np.sqrt(np.var((chm).flatten()))		# adjust the variance
			chm = chm + _mean				# adjust the mean
	
			# test
#			np.savetxt('chm.csv', im, delimiter=',')
#			_dst = np.sqrt(np.sum((im - chm)**2))
#			LOGGER.debug('change {}'.format(str(_dst)))
	else:
		chm = im

	return chm
	


'''
	modify kurtosis

	this is a port of textureSynth/modkurt.m by J. Portilla and E. Simoncelli.
	https://github.com/LabForComputationalVision/textureSynth

	http://www.cns.nyu.edu/~lcv/texture/

'''
def mod_kurt(im, kt):
	# mu
	_mean = np.mean(im.flatten())
#	_sd = np.sqrt(np.var(im.flatten()))

	im = im - _mean

	_tmp = im**2
	_sd = np.sqrt(np.mean(_tmp.flatten()))

	mu = [im**i for i in range(3,13)]
	mu = [np.mean(mu[i].flatten()) for i in range(len(mu))]
	
	if _sd > 0:
		_kt = mu[1] / (_sd)**4

		_a = mu[1] / _sd**2

		_A = mu[9] - 4.*_a*mu[7] - 4*mu[0]*mu[6] + 6.*_a**2*mu[5] + 12*_a*mu[0]*mu[4] + 6*mu[0]**2*mu[3] \
			- 4*_a**3*mu[3] - 12*_a**2*mu[0]*mu[2] + _a**4*mu[1] - 12*_a*mu[0]**2*mu[1] \
			+ 4*_a**3*mu[0]**2 + 6*_a**2*mu[0]**2*_sd**2 - 3.*mu[0]**4
		_B = 4.* ( mu[7] - 3*_a*mu[5] - 3*mu[0]*mu[4] + 3.*_a**2*mu[3] + 6.*_a*mu[0]*mu[2] + 3.*mu[0]**2*mu[1] \
			- _a**3*mu[1] - 3.*_a**2*mu[0]**2 - 3*mu[1]*mu[0]**2 )
		_C = 6.* ( mu[5] - 2.*_a*mu[3] - 2.*mu[0]*mu[2] + _a**2*mu[1] + 2.*_a*mu[0]**2 + mu[0]**2*_sd**2 )
		_D = 4.* ( mu[3] - _a**2*_sd**2 - mu[0]**2 )
		_E = mu[1]

		# Define the coefficients of the denominator (F*lam^2+G)^2
		_F = _D / 4.
		_G = _sd**2

		d = np.zeros_like(range(0,5), dtype='double')
		d[0] = _B * _F
		d[1] = 2.*_C*_F - 4.*_A*_G
		d[2] = 4.*_F*_D - 3.*_B*_G - _D*_F
		d[3] = 4.*_F*_E - 2.*_C*_G
		d[4] = -1. * _D * _G

		mMlambda = np.roots(d)

		tg = mMlambda.imag / mMlambda.real
		_idx = np.where(np.abs(tg) < 1e-6)
		mMlambda = mMlambda[_idx].real
		lNeg = mMlambda[np.where(mMlambda < 0)]
		if lNeg.shape[0] == 0:
			lNeg = -1/2**-50

		lPos = mMlambda[np.where(mMlambda >= 0)]
		if lPos.shape[0] == 0:
			lPos = 1/2**-50

		lmi = np.max(lNeg)
		lma = np.min(lPos)

		lam = np.array([lmi, lma], dtype='double')

		mMnewKt = np.polyval(np.array([_A, _B, _C, _D, _E], dtype='double'), lam) / np.polyval(np.array([_F, 0, _G], dtype='double'), lam)**2

		kmin = np.min(mMnewKt)
		kmax = np.max(mMnewKt)

	# Given a desired skewness, solves for lambda
		if kt <= kmin:
			lamb = lmi
			LOGGER.debug('Saturating (down) skewness!')
		elif kt >= kmax:
			lamb = lma
			LOGGER.debug('Saturating (up) skewness!')
		else:
			c = np.zeros_like(range(0,5), dtype='double')
			_tmp = kt*(_G**2)
			c[0] = _E - _tmp
			c[1] = _D
			c[2] = _C - 2.*kt*_F*_G
			c[3] = _B
			c[4] = _A - kt*_F**2

			c = c[::-1]

			r = np.roots(c)

# Chose the real solution with minimum absolute value with the rigth sign
			lam = np.array( [0.] )
			co = 0
			tg = r.imag / r.real
			_idx = np.where( np.abs(tg) == 0. )
			lam = r[_idx].real

			if lam.shape[0] > 0:
				lamb = lam[np.where(np.abs(lam) == np.min(np.abs(lam)))].real
				lamb = lamb[0]
			else:
				lamb = 0.

		# Modify the channel
		chm = im + lamb*(im**3 - _a*im - mu[0])		# adjust the skewness
		chm = chm * _sd / np.sqrt(np.var((chm).flatten()))		# adjust the variance
		chm = chm + _mean				# adjust the mean

#		_dst = np.sqrt(np.sum((im - chm)**2))
#		LOGGER.debug('change {}'.format(str(_dst)))

	else:
		chm = im

	return chm


'''
	modify auto correlation

	this is a port of textureSynth/modacor22.m by J. Portilla and E. Simoncelli.
	https://github.com/LabForComputationalVision/textureSynth
	
	http://www.cns.nyu.edu/~lcv/texture/

'''
def mod_acorr(im, cy, mm):
	_la = np.floor((mm-1)/2)
	_nc = cy.shape[1]

	centy = int(im.shape[0]/2+1)
	centx = int(im.shape[1]/2+1)

	# calicurate auto correlation of original image.
	ft = np.fft.fft2(im)
	ft2 = np.abs(ft)**2
	cx = np.fft.ifftshift(np.fft.ifft2(ft2).real)
	if not np.all(np.isreal(cx)):
		cx = cx / 2.
	
	cy = cy*np.prod(im.shape)	# Unnormalize the previously normalized correlation

	# Take just the part that has influence on the samples of cy (cy=conv(cx,im))
	ny = int(cx.shape[0]/2.0)+1
	nx = int(cx.shape[1]/2.0)+1
	_sch = min((ny, nx))
	le = int(min((_sch/2-1, _la)))

	cx = cx[ny-2*le-1: ny+2*le, nx-2*le-1: nx+2*le]

	# Build the matrix that performs the convolution Cy1=Tcx*Ch1
	_ncx = 4*le + 1
	_win = int(((_nc)**2 + 1)/2)
	_tcx = np.zeros((_win, _win))

	for i in range(le+1, 2*le+1):
		for j in range(le+1, 3*le+2):
			ccx = cx[i-le-1:i+le, j-le-1:j+le].copy()
			ccxi = ccx[::-1, ::-1]
			ccx += ccxi
			ccx[le, le] = ccx[le, le]/2.
			ccx = ccx.flatten()
			nm = (i-le-1)*(2*le+1) + (j-le) 
			_tcx[nm-1,] = ccx[0:_win]

	i = 2*le + 1
	for j in range(le+1, 2*le+2):
		ccx = cx[i-le-1:i+le, j-le-1:j+le].copy()
		ccxi = ccx[::-1, ::-1]
		ccx = ccx + ccxi
		ccx[le, le] = ccx[le, le]/2.
		ccx = ccx.flatten()
		nm = (i-le-1)*(2*le+1) + (j-le)
		_tcx[nm-1,] = ccx[0:_win]

	# Rearrange Cy indices and solve the equation
	cy1 = cy.flatten()
	cy1 = cy1[0:_win]

	# np.solve might be better than np.inv
	ch1 = np.linalg.solve(_tcx, cy1)

	# Rearrange Ch1

	ch1 = np.hstack((ch1, ch1[-2::-1]))
	ch = ch1.reshape((_nc, _nc))

	aux = np.zeros(im.shape)
	aux[centy-le-1:centy+le, centx-le-1:centx+le] = ch
	ch = np.fft.fftshift(aux)
	chf = np.fft.fft2(ch).real

	yf = ft*np.sqrt(np.abs(chf))
	y = np.fft.ifft2(yf).real

	return y


'''
	adjust correlation

	this is a port of textureSynth/adjustCorr1s.m by J. Portilla and E. Simoncelli.
	https://github.com/LabForComputationalVision/textureSynth
	
	http://www.cns.nyu.edu/~lcv/texture/

'''
def adjust_corr1(xx, c0):

	# get variance
	_C = np.dot(xx.T, xx) / xx.shape[0]
	_D, _E = np.linalg.eig(_C)

	_D[np.where(np.abs(_D) < PS)] = 0
	if np.sum(np.where(_D < 0)):
		LOGGER.info('negative eigenvalue')
		LOGGER.info(_D)
		LOGGER.info(_C)
	_idx = np.argsort(_D)[::-1]
	_D = np.diag(np.sqrt(_D[_idx]))
	_iD = np.zeros_like(_D)
	_iD[np.where(_D != 0.)] = 1. / _D[np.where(_D != 0.)]
	_E = _E[:, _idx]

	_D0, _E0 = np.linalg.eig(c0)

	_D0[np.where(np.abs(_D0) < PS)] = 0
	if np.sum(np.where(_D0 < 0)):
		LOGGER.info('negative eigenvalue')
		LOGGER.info(_D0)
		LOGGER.info(c0)
		LOGGER.info(c0-c0.T)

	_idx = np.argsort(_D0)[::-1]
	_D0 = np.diag(np.sqrt(_D0[_idx]))
	_E0 = _E0[:, _idx]

	_orth = np.dot(_E.T, _E0)

	# _E * inv(D) * _orth * _D0 * _E0'
	_M = np.dot(_E, np.dot(_iD, np.dot(_orth, np.dot(_D0, _E0.T))))

	_new = np.dot(xx, _M)

	return _new


'''
	adjust correlation

	this is a port of textureSynth/adjustCorr2s.m by J. Portilla and E. Simoncelli.
	https://github.com/LabForComputationalVision/textureSynth

	http://www.cns.nyu.edu/~lcv/texture/

'''
def adjust_corr2(xx, cx, yy, cxy):
	# subtract mean
	_mean = np.mean(xx, axis=0)
	xx = xx - _mean
	_mean = np.mean(yy, axis=0)
	yy = yy - _mean
	# get variance , covariance
	_Bx = np.dot(xx.T, xx) / xx.shape[0]
	_Bxy = np.dot(xx.T, yy) / xx.shape[0]
	_By = np.dot(yy.T, yy) / yy.shape[0]
	_iBy = np.linalg.inv(_By)

	_Cur = _Bx - np.dot(_Bxy, np.dot(_iBy, _Bxy.T))
	_Des = cx - np.dot(cxy, np.dot(_iBy, cxy.T))

	_D, _E = np.linalg.eig(_Cur)
	_D[np.where(np.abs(_D) < PS)] = 0
	if np.sum(np.where(_D < 0)):
		LOGGER.info('negative eigenvalue')
		LOGGER.info(_D)
	_idx = np.argsort(_D)[::-1]
	_D = np.diag(np.sqrt(_D[_idx]))
	_iD = np.zeros_like(_D)
	_iD[np.where(_D != 0.)] = 1. / _D[np.where(_D != 0.)]
	_E = _E[:, _idx]

	_D0, _E0 = np.linalg.eig(_Des)
	_D0[np.where(np.abs(_D0) < PS)] = 0
	if np.sum(np.where(_D0 < 0)):
		LOGGER.info('negative eigenvalue')
		LOGGER.info(_D0)
	_idx = np.argsort(_D0)[::-1]
	_D0 = np.diag(np.sqrt(_D0[_idx]))
	_E0 = _E0[:, _idx]

	_orth = np.dot(_E.T, _E0)

	# _E * inv(D) * _orth * _D0 * _E0'
	_Mx = np.dot(_E, np.dot(_iD, np.dot(_orth, np.dot(_D0, _E0.T))))

	_My = np.dot(_iBy, (cxy.T - np.dot(_Bxy.T, _Mx)))

	_new = np.dot(xx, _Mx) + np.dot(yy, _My)
	
	return _new


'''
	calicurate auto correlation

'''
def get_acorr(im, mm):
	_fr = np.fft.fft2(im)
#	_fr = np.fft.fftshift(np.fft.fft2(im))
	_la = np.floor((mm-1)/2)
	
	_t = np.absolute(_fr)
	_tmp = _t ** 2 / np.prod(_t.shape)
#	_tmp = ( _t - np.mean(_t.flatten()) )**2 / np.prod(_t.shape)
	# important!! auto-correlation
	_tmp = np.fft.ifft2(_tmp)
	_tmp = _tmp.real
	_tmp = np.fft.ifftshift(_tmp)
#	_tmp = np.absolute(_tmp)

	ny = int(_t.shape[0]/2.0)
	nx = int(_t.shape[1]/2.0)
	
	_sch = min((ny, nx))
	le = int(min((_sch/2-1, _la)))
	ac = _tmp[ny-le: ny+le+1, nx-le: nx+le+1]

	return ac


'''
	covariance matrix of color image(3 channels)

'''
def cov_im(im):
	_tmp = np.array(im)

	_list = np.zeros((_tmp.shape[0]*_tmp.shape[1], _tmp.shape[2]))

	_dp = []
	for i in range(_tmp.shape[2]):
		_list[:, i] = _tmp[:, :, i].flatten()

	_mean = np.mean(_list, axis=0)
	_list -= _mean

	_t = np.dot(_list.T, _list) / _list.shape[0]

	return _t


'''
	means of color image(3 channels)

'''
def mean_im(im):
	_tmp = np.array(im)

	_list = np.zeros((_tmp.shape[0]*_tmp.shape[1], _tmp.shape[2]))

	_dp = []
	for i in range(_tmp.shape[2]):
		_list[:, i] = _tmp[:, :, i].flatten()

	_mean = np.mean(_list, axis=0)

	return _mean


'''
	normalized PCA

'''
def get_pca_test(image):
	# reshape to ['width of _img' * 'height', 'channel'] matrix.
	_img = image.reshape(image.shape[0]*image.shape[1], image.shape[2])

	pca = PCA()
	pca.fit(_img)

	_pcdata = pca.transform(_img)

	# normalize _pcdate
	_sd = np.sqrt(np.var(_pcdata, axis=0))
	_pcdata = _pcdata / _sd

	_pcdata = _pcdata.reshape(image.shape[0], image.shape[1], image.shape[2])

	return _pcdata

'''
	normalized PCA

'''
def get_pca(image):

	# reshape to ['width of _img' * 'height', 'channel'] matrix.
	_img = image.reshape(image.shape[0]*image.shape[1], image.shape[2])

	_mean = np.mean(_img, axis=0)
	_tmp = _img - _mean
	_covar = np.dot(_tmp.T, _tmp)/_img.shape[0]

	_eval, _evec = np.linalg.eig(_covar)
	_idx = np.argsort(_eval)[::-1]
	_ediag = np.diag(_eval[_idx])
	_evec = _evec[:, _idx]
	## this treatment is to get same results as Matlab
	for k in range(_evec.shape[1]):
		if np.sum(_evec[:,k] < 0) > np.sum(_evec[:,k] >= 0):
			_evec[:,k] = -1. * _evec[:,k]

	# get principal components
	_pcscore = np.dot(_tmp, _evec)

	# Moore-Penrose Pseudo Inverse.
	## Generalized inverse matrix is not necessary for this case. (trivial)
	## [Attn.] Bellow (1/4 power) may be mistake of textureColorAnalysis.m/textureColorSynthesis.m.
	## **(0.5) would be right. this obstructs color reproduction.
	#_iediag = np.linalg.pinv(_ediag**(0.25))
	_iediag = np.linalg.pinv(_ediag**(0.5))
		
	# normalize principal components
	_npcdata = np.dot(_pcscore, _iediag)
	_npcdata = _npcdata.reshape(image.shape[0], image.shape[1], image.shape[2])

	return _npcdata


'''
	(1) marginal statistics
		mean, variance, skewness, kurtosis, range of original image
		variance of highpass residual		
'''
def mrg_stats(image):
		
	_mean = np.mean(image.real.flatten())
	_var = np.var(image.real.flatten())
	_skew = skew(image.real.flatten())
	_kurt = kurtosis(image.real.flatten()) + 3.0 # make same as MATLAB
	_max = np.max(image.real)
	_min = np.min(image.real)

	return [ _mean, _var, _skew, _kurt, _max, _min ]


'''
	auto-correlation of lowpass residual (Color Version)

'''
def cov_lr(lores):
	
	_dim = 4 * lores[0]['s'].shape[0] * lores[0]['s'].shape[1]

	# expand residuals and combine slided vectors
	_vec = get_2slide(lores)

#	_mean = np.mean(_vec, axis=0)
#	_vec -= _mean

	_res = np.dot(_vec.T, _vec) / _dim
#
	return _res

'''
	conbine slided residuals (Color Version)
'''
def get_2slide(lores):
	_dim = lores[0]['s'].shape[0] * lores[0]['s'].shape[1]
	_dim = 4 * lores[0]['s'].shape[0] * lores[0]['s'].shape[1]
	_vec = np.zeros((_dim, 15))

	for i in range(len(lores)):
		_lo = expand(lores[i]['s'], 2, 1) / 4
		_lo = _lo.real
		_vec[:, 0 + 5*i] = _lo.reshape(-1,)
#		_vec[:, 0 + 5*i] = _lo.flatten()
		_vec[:, 1 + 5*i] = np.roll(_lo, 2, axis=0).flatten()
		_vec[:, 2 + 5*i] = np.roll(_lo, -2, axis=0).flatten()
		_vec[:, 3 + 5*i] = np.roll(_lo, 2, axis=1).flatten()
		_vec[:, 4 + 5*i] = np.roll(_lo, -2, axis=1).flatten()

	return _vec



'''
	auto-correlation of lowpass residual (Gray Version)

'''
def cov_lr_g(lores):
	
	_dim = 4 * lores['s'].shape[0] * lores['s'].shape[1]

	# expand residuals and combine slided vectors
	_vec = get_2slide_g(lores)


	_res = np.dot(_vec.T, _vec) / _dim

	return _res


'''
	conbine slided residuals (Gary Version)
'''
def get_2slide_g(lores):
	_dim = lores['s'].shape[0] * lores['s'].shape[1]
	_dim = 4 * lores['s'].shape[0] * lores['s'].shape[1]
	_vec = np.zeros((_dim, 15))

	_lo = expand(lores['s'], 2, 1) / 4
	_lo = _lo.real
	_vec[:, 0] = _lo.reshape(-1,)
	_vec[:, 1] = np.roll(_lo, 2, axis=0).flatten()
	_vec[:, 2] = np.roll(_lo, -2, axis=0).flatten()
	_vec[:, 3] = np.roll(_lo, 2, axis=1).flatten()
	_vec[:, 4] = np.roll(_lo, -2, axis=1).flatten()

	return _vec



'''
	get magnitude and real values of bandpass

'''
def trans_b(b):
	b_m = []
	for i in range(len(b)):
		_tmp = []
		for j in range(len(b[i])):
			_tmp.append(np.abs(b[i][j]['s']))
		b_m.append(_tmp)

	b_r = []
	for i in range(len(b)):
		_tmp = []
		for j in range(len(b[i])):
			_tmp.append(b[i][j]['s'].real)
		b_r.append(_tmp)

	b_i = []
	for i in range(len(b)):
		_tmp = []
		for j in range(len(b[i])):
			_tmp.append(b[i][j]['s'].imag)
		b_i.append(_tmp)

	return b_m, b_r, b_i


'''
	get parents of bandpass (Color)

'''
def get_parent(b, lores):
	b_p = []
	b_rp = []
	b_ip = []

	for i in range(len(b)):
		if i < len(b) - 1:
			_dimy = b[i][0]['s'].shape[0] * b[i][0]['s'].shape[1]
			_p = np.zeros((_dimy, len(b[i])))
			_rp = np.zeros_like(_p)
			_ip = np.zeros_like(_p)
			for j in range(len(b[i])):
				# expand parent bandpass
				_tmp = expand(b[i+1][j]['s'], 2) / 4.
				# double phase
				_tmp = double_phase(_tmp).flatten()
				_p[:, j] = np.abs(_tmp) # magitude
				_rp[:, j] = _tmp.real # real value
				_ip[:, j] = _tmp.imag # imaginary value

			_p -= np.mean(_p, axis=0)

			b_p.append(_p)
			b_rp.append(_rp)
			b_ip.append(_ip)

		else:
			# when no parents
			_tmp = expand(lores['s'], 2).real / 4.
			_dimy = _tmp.shape[0] * _tmp.shape[1]
			_rp = np.zeros((_dimy, 5))
			_rp[:, 0] = _tmp.flatten()
			_rp[:, 1] = np.roll(_tmp, 2, axis=1).flatten()
			_rp[:, 2] = np.roll(_tmp, -2, axis=1).flatten()
			_rp[:, 3] = np.roll(_tmp, 2, axis=0).flatten()
			_rp[:, 4] = np.roll(_tmp, -2, axis=0).flatten()
			b_rp.append(_rp)

	return b_p, b_rp, b_ip


'''
	get parents of bandpass (Gray)

'''
def get_parent_g(b, lores):
	b_p = []
	b_rp = []
	b_ip = []

	for i in range(len(b)-1):
		_dimy = b[i][0]['s'].shape[0] * b[i][0]['s'].shape[1]
		_p = np.zeros((_dimy, len(b[i])))
		_rp = np.zeros_like(_p)
		_ip = np.zeros_like(_p)
		for j in range(len(b[i])):
			# expand parent bandpass
			_tmp = expand(b[i+1][j]['s'], 2) / 4.
			# double phase
			_tmp = double_phase(_tmp).flatten()
			_p[:, j] = np.abs(_tmp) # magitude
			_rp[:, j] = _tmp.real # real value
			_ip[:, j] = _tmp.imag # imaginary value

		_p -= np.mean(_p, axis=0)

		b_p.append(_p)
		b_rp.append(_rp)
		b_ip.append(_ip)

	return b_p, b_rp, b_ip


'''
	central auto-correlation of magnitude of bandpass

'''
def autocorr_b(b, MM):
	b_c = []
	for i in range(len(b)):
		_tmp = []
		for j in range(len(b[i])):
			_tmp.append(get_acorr(b[i][j], MM))
		b_c.append(_tmp)

	return b_c



'''
	marginal statistics of magnitude of bandpass

'''
def mrg_b(b):
	b_c = []
	for i in range(len(b)):
		_tmp = []
		for j in range(len(b[i])):
			_tmp.append(mrg_stats(b[i][j]))
		b_c.append(_tmp)

	return b_c


'''
	combine colors	(color version)

'''
def cclr_b(bnd, dp):
	if len(bnd[0]) < dp:
		return np.array([])

	_tmp0 = bnd[0][dp]
	_tmp1 = bnd[1][dp]
	_tmp2 = bnd[2][dp]

	_ori = len(_tmp0)
	_dy = _tmp0[0].shape[0]
	_dx = _tmp0[0].shape[1]
	_list = np.zeros((_dy*_dx, 3*_ori))
	for j in range(_ori):
		_list[:, j] = _tmp0[j].flatten()
		_list[:, _ori+j] = _tmp1[j].flatten()
		_list[:, 2*_ori+j] = _tmp2[j].flatten()

	return _list

def cclr_bc(bnd, dp):
	if len(bnd[0]) < dp:
		return np.array([])

	_tmp0 = bnd[0]
	_tmp1 = bnd[1]
	_tmp2 = bnd[2]

	_ori = len(_tmp0)
	_dy = _tmp0[0].shape[0]
	_dx = _tmp0[0].shape[1]
	_list = np.zeros((_dy*_dx, 3*_ori))
	for j in range(_ori):
		_list[:, j] = _tmp0[j].flatten()
		_list[:, _ori+j] = _tmp1[j].flatten()
		_list[:, 2*_ori+j] = _tmp2[j].flatten()

	return _list


def cclr_p(bnd, dp):
	if len(bnd[0]) < dp:
		return np.array([])

	_tmp0 = bnd[0][dp]
	_tmp1 = bnd[1][dp]
	_tmp2 = bnd[2][dp]

	_ori = 	_tmp0.shape[1]
	_dy = _tmp0.shape[0]
	_list = np.zeros((_dy, 3*_ori))
	for j in range(_ori):
		_list[:, j] = _tmp0[:, j]
		_list[:, _ori+j] = _tmp1[:, j]
		_list[:, 2*_ori+j] = _tmp2[:, j]

	return _list


def cclr_rp(bnd, bnd_i, dp):
	if len(bnd[0]) < dp:
		return np.array([])

	_tmp0 = bnd[0][dp]
	_tmp1 = bnd[1][dp]
	_tmp2 = bnd[2][dp]
	_ori = 	_tmp0.shape[1]
	_dy = _tmp0.shape[0]
	
	if len(bnd_i) < dp:
		_list = np.zeros((_dy, 3*_ori))
		for j in range(_ori):
			_list[:, j] = _tmp0[:, j]
			_list[:, _ori+j] = _tmp1[:, j]
			_list[:, 2*_ori+j] = _tmp2[:, j]
	else:
		_tmp3 = bnd_i[0][dp]
		_tmp4 = bnd_i[1][dp]
		_tmp5 = bnd_i[2][dp]
		_list = np.zeros((_dy, 6*_ori))
		for j in range(_ori):
			_list[:, j] = _tmp0[:, j]
			_list[:, _ori+j] = _tmp1[:, j]
			_list[:, 2*_ori+j] = _tmp2[:, j]
			_list[:, 3*_ori+j] = _tmp3[:, j]
			_list[:, 4*_ori+j] = _tmp4[:, j]
			_list[:, 5*_ori+j] = _tmp5[:, j]

	return _list


'''
	separate colors	

'''
def sclr_b(cous, ori):

	_dim = ( int(np.sqrt(cous.shape[0])), int(cous.shape[1]/3) )
	_ori = _dim[1]

	_list = []
	for i in range(3):
		_tmp = np.zeros((_dim[0], _dim[0]))
		_vec = []

		for j in range(_ori):
			_tmp = cous[:, _ori*i +j].reshape((_dim[0], _dim[0]))
			_vec.append(_tmp)
		
		_list.append(_vec)

	return _list



'''
	combine orientations (Gray Only)

'''
def cori_b(bnd, dp):
	if len(bnd) < dp:
		return np.array([])

	_tmp0 = bnd[dp]

	_ori = len(_tmp0)
	_dy = _tmp0[0].shape[0]
	_dx = _tmp0[0].shape[1]
	_list = np.zeros((_dy*_dx, _ori))
	for j in range(_ori):
		_list[:, j] = _tmp0[j].flatten()

	return _list


def cori_bc(bnd, dp):
	if len(bnd) < dp:
		return np.array([])

	_tmp0 = bnd

	_ori = len(_tmp0)
	_dy = _tmp0[0].shape[0]
	_dx = _tmp0[0].shape[1]
	_list = np.zeros((_dy*_dx, _ori))
	for j in range(_ori):
		_list[:, j] = _tmp0[j].flatten()

	return _list


def cori_rp(bnd, bnd_i, dp):
	if len(bnd[0]) < dp:
		return np.array([])

	_tmp0 = bnd[dp]
	_tmp1 = bnd_i[dp]
	_ori = 	_tmp0.shape[1]
	_dy = _tmp0.shape[0]
	
	_list = np.zeros((_dy, 2*_ori))
	for j in range(_ori):
		_list[:, j] = _tmp0[:, j]
		_list[:, _ori+j] = _tmp1[:, j]

	return _list


'''
	separate orientations (Gray ony)	

'''
def sori_b(cous, ori):

	_dim = ( int(np.sqrt(cous.shape[0])), int(cous.shape[1]) )
	_ori = _dim[1]

	_list = []

	for j in range(_ori):
		_tmp = cous[:, j].reshape((_dim[0], _dim[0]))
		_list.append(_tmp)

	return _list

'''
	Create mirrored image

'''
def pad_reflect(image):
	image1 = np.pad(image, [[int(image.shape[0]/2), int(image.shape[0]/2)], [int(image.shape[1]/2), int(image.shape[1]/2)]] , 'reflect')
	return image1


