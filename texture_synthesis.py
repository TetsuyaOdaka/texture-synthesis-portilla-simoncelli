'''
	Texture Synthesis  (Color Version)

	this is a port of textureSynth/textureColorSynthesis.m by J. Portilla and E. Simoncelli.
	http://www.cns.nyu.edu/~lcv/texture/

	Differences:
	(1) i use real version of steerable pyramid.
	    Sorry. I could not understand the algorithm of complex version in textureSynthesis.m.
	(2) i don't use filter masks of orientations in the process of coarse to fine.

	Usage:
	python texture_synthesis.py -i pebbles.jpg -o tmp -n 5 -k 4 -m 7 --iter 100

	-i : input image path
	-o : path for output
	-n : depth of steerable pyramid (default:5)
	-k : num of orientations of steerable pyramid (default:4)
	-n : pixel distance for calicurationg auto-correlations (default:7)
	--iter : number of iterations (default:100)


'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import LinAlgError
from scipy.stats import skew, kurtosis
from PIL import Image
import sys, os
import logging
import argparse, copy
import time

import sutils
import steerable_pyramid as steerable
import texture_analysis as ta

SCRIPT_NAME = os.path.basename(__file__)

# logging
LOG_FMT = "[%(name)s] %(asctime)s %(levelname)s %(lineno)s %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FMT)
LOGGER = logging.getLogger(os.path.basename(__file__))


'''
	Texture Synthesis by Portilla-Simoncelli's algorithm

'''
def synthesis(image, resol_x, resol_y, num_depth, num_ori, num_neighbor, iter, out_path):
	# analyse original image
	orig_data = ta.TextureAnalysis(image, resol_x, resol_y, num_depth, num_ori, num_neighbor)
	orig_data.analyse()

	# initialize random image
	im = np.random.normal(0, 1, resol_x * resol_y * 3).reshape(resol_y*resol_x, 3)
# test
#	im = np.loadtxt("init-im.csv",delimiter=",")
	## adjust covariance among RGB channels
	_tmp = orig_data.COV_RGB
	im = sutils.adjust_corr1(im, _tmp)
	im = im + orig_data.MEAN_RGB
	im = im.reshape(resol_y, resol_x, 3)

	# eigen values and vectors of original image
	ocov_eval, ocov_evec = np.linalg.eig(orig_data.COV_RGB)
	_idx = np.argsort(ocov_eval)[::-1]
	ocov_ediag = np.diag(ocov_eval[_idx])
	ocov_evec = ocov_evec[:, _idx]
	## this treatment is to get same result as Matlab
	for k in range(ocov_evec.shape[1]):
		if np.sum(ocov_evec[:,k] < 0) > np.sum(ocov_evec[:,k] >= 0):
			ocov_evec[:,k] = -1. * ocov_evec[:,k]
	## [Attn.] Bellow (1/4 power) may be mistake of textureColorAnalysis.m/textureColorSynthesis.m.
	## **(0.5) would be right. this obstructs color reproduction.
#	ocov_iediag = np.linalg.pinv(ocov_ediag**(0.25))
	# Moore-Penrose Pseudo Inverse.
	ocov_iediag = np.linalg.pinv(ocov_ediag**(0.5))
	
	# iteration
	prev_im = np.array([])
	prev_dst = 0.

	for it in range(0, iter):
		LOGGER.debug('iteration {}'.format(str(it)))

		# ------------------------------------
		# Normalized pca components
		_dim = im.shape
		im = im.reshape(_dim[0]*_dim[1], 3)
		_mean = np.mean(im, axis=0)
		im = im - _mean
		## get principal components
		_pcscore = np.dot(im, ocov_evec)
		## normalize principal components 
		im = np.dot(_pcscore, ocov_iediag)
		im = im.reshape(_dim[0], _dim[1], 3)
	
		pyr_l = []
		lr_l = []

		# ------------------------------------
		# Create pyramids of each PCA channel
		for clr in range(3):
			# steerable pyramid
			_sp = steerable.SteerablePyramid(im[:, :, clr], resol_x, resol_y, num_depth, num_ori, '', '', 0)
			_sp.create_pyramids()

			# subtract means from lowpass residuals
			_sp.LR['s'] = _sp.LR['s'].real - np.mean(_sp.LR['s'].real.flatten())

			pyr_l.append(copy.deepcopy(_sp))
			lr_l.append(_sp.LR)

		# ------------------------------------
		# Adjust lowpass residual and get initial image for coarse to fine
		## get auto-correlations (2 slide)
		## this tric is according to textureSynthesis.m
		_mat = sutils.get_2slide(lr_l)

		## adjust auto correlation of lowpass residuals
		_mat = sutils.adjust_corr1(_mat, orig_data.COV_LR)

		## back to lowpass residuals
		_dim = tuple(map(lambda x: x * 2, _sp.LR['s'].shape))
		for clr in range(3):
			_tns = np.zeros((_dim[0], _dim[1], 5))
			_tns[:, :, 0] = _mat[:, 0 + 5*clr].reshape(_dim[0], _dim[1])
			_tns[:, :, 1] = np.roll(_mat[:, 1 + 5*clr].reshape(_dim[0], _dim[1]), -2, axis=1)
			_tns[:, :, 2] = np.roll(_mat[:, 2 + 5*clr].reshape(_dim[0], _dim[1]), 2, axis=1)
			_tns[:, :, 3] = np.roll(_mat[:, 3 + 5*clr].reshape(_dim[0], _dim[1]), -2, axis=0)
			_tns[:, :, 4] = np.roll(_mat[:, 4 + 5*clr].reshape(_dim[0], _dim[1]), 2, axis=0)
			_mean = np.mean(_tns, axis=2)
			_mean = sutils.shrink(_mean, 2) * 4.
			lr_l[clr]['s'] = _mean
			lr_l[clr]['f'] = np.fft.fftshift(np.fft.fft2(_mean))
			pyr_l[clr].LR['s'] = lr_l[clr]['s']
			pyr_l[clr].LR['f'] = lr_l[clr]['f']

		## get initial data for coarse to fine
		rec_im = []
		for clr in range(3):
			# get lowband
			_z = np.zeros_like(lr_l[clr]['f'])
			_s = steerable.SteerablePyramid(_z, _z.shape[1], _z.shape[0], 1, num_ori, '', '', 0)
			_lr_f = lr_l[clr]['f'] * _s.L0_FILT
			_lr_s = np.fft.ifft2(np.fft.ifftshift(_lr_f)).real
			# modify central auto correlation
			if(orig_data.LR_MAR[clr][1]/ocov_ediag[clr,clr] > 1.0e-3):
				try:
					lr_l[clr]['s'] = sutils.mod_acorr(_lr_s, orig_data.LR_CA[clr], num_neighbor)
				except LinAlgError as e:
					LOGGER.info('LinAlgError {}'.format(e))
			else:
				lr_l[clr]['s'] = lr_l[clr]['s'] * np.sqrt(orig_data.LR_MAR[clr][1] / np.var(lr_l[clr]['s']))

			lr_l[clr]['s'] = lr_l[clr]['s'].real
			# modify skewness of lowpass residual
			lr_l[clr]['s'] = sutils.mod_skew(lr_l[clr]['s'], orig_data.LR_MAR[clr][2])
			# modify kurtosis of lowpass residual
			lr_l[clr]['s'] = sutils.mod_kurt(lr_l[clr]['s'], orig_data.LR_MAR[clr][3])
			lr_l[clr]['f'] = np.fft.fftshift(np.fft.fft2(lr_l[clr]['s']))

			 # initial coarse to fine
			rec_im.append(lr_l[clr]['s'])

		## get original statistics of bandpass signals.
		bnd = []
		bnd_m = []
		bnd_p = []
		bnd_rp = []
		bnd_ip = []
		for clr in range(3):
			# create parents
			bnd.append(copy.deepcopy(pyr_l[clr].BND))
			_b_m, _, _ = sutils.trans_b(pyr_l[clr].BND)
			for i in range(len(_b_m)):
				for k in range(len(_b_m[i])):
					_b_m[i][k] -= np.mean(_b_m[i][k])
			## magnitude
			bnd_m.append(_b_m)

			_b_p, _b_rp, _b_ip = sutils.get_parent(pyr_l[clr].BND, pyr_l[clr].LR)
			## maginitude of parent bandpass  (this is 'parent' in textureColorAnalysis.m)
			bnd_p.append(_b_p)
			## real values of parent bandpass (this is half of 'rparent' in textureColorAnalysis.m)
			bnd_rp.append(_b_rp)
			## imaginary values of parent bandpass (this is half of 'rparent' in textureColorAnalysis.m)
			bnd_ip.append(_b_ip)

		# ------------------------------------
		# Coarse to fine adjustment
		for dp in range(num_depth-1, -1, -1):

			# combine colors
			cousins = sutils.cclr_b(bnd_m, dp)
			rparents = sutils.cclr_rp(bnd_rp, bnd_ip, dp)

			# adjust covariances
			_prev = cousins
			if dp < num_depth-1:
				parents = sutils.cclr_p(bnd_p, dp)
				cousins = sutils.adjust_corr2(_prev, orig_data.CF_COUS[dp], parents, orig_data.CF_CPAR[dp])
				if np.isnan(cousins).any():
					LOGGER.info('NaN in adjust_corr2')
					cousins = sutils.adjust_corr1(_prev, orig_data.CF_COUS[dp])
			else:
				cousins = sutils.adjust_corr1(_prev, orig_data.CF_COUS[dp])

			# separate colors
			cousins = sutils.sclr_b(cousins, num_ori)

			# adjust central auto corr. and update bandpass.
			bnd_r = []
			for clr in range(3):
				_list = []
				for k in range(num_ori):
					# adjust central auto-correlations
					_tmp = sutils.mod_acorr(cousins[clr][k], orig_data.BND_MCOR[clr][dp][k], num_neighbor)
					# update BND_N
					bnd_m[clr][dp][k] = _tmp
					_mean = orig_data.BND_MMAR[clr][dp][k][0]
					_tmp = _tmp + _mean
					_idx = np.where(_tmp < 0)
					_tmp[_idx] = 0

					_bnd = pyr_l[clr].BND[dp][k]['s']
					_idx1 = np.where(np.abs(_bnd) < 10**(-12))
					_idx2 = np.where(np.abs(_bnd) >= 10**(-12))
					_bnd[_idx1] = _bnd[_idx1] * _tmp[_idx1]
					_bnd[_idx2] = _bnd[_idx2] * _tmp[_idx2] / np.abs(_bnd[_idx2])

					_list.append(_bnd.real)

				bnd_r.append(_list)

			# combine colors & make rcousins
			rcousins = sutils.cclr_bc(bnd_r, dp)

			# adjust cross-correlation of real values of B and real/imaginary values of parents
			_prev = rcousins
			try:
				rcousins = sutils.adjust_corr2(_prev, orig_data.CF_RCOU[dp], rparents, orig_data.CF_RPAR[dp])
				if np.isnan(rcousins).any():
					LOGGER.info('NaN in adjust_corr2')
					rcousins = sutils.adjust_corr1(_prev, orig_data.CF_RCOU[dp])
					if np.isnan(rcousins).any():
						LOGGER.info('NaN in adjust_corr1')
						rcousins = _prev
			except LinAlgError as e:
				LOGGER.info('LinAlgError {}'.format(e))
				rcousins = sutils.adjust_corr1(_prev, orig_data.CF_RCOU[dp])
				if np.isnan(rcousins).any():
					LOGGER.info('NaN in adjust_corr1')
					rcousins = _prev

			# separate colors
			rcousins = sutils.sclr_b(rcousins, num_ori)
			for clr in range(3):
				for k in range(num_ori):
					## update pyramid
					pyr_l[clr].BND[dp][k]['s'] = rcousins[clr][k]
					pyr_l[clr].BND[dp][k]['f'] = np.fft.fftshift(np.fft.fft2(rcousins[clr][k]))

			# combine bands
			_rc = copy.deepcopy(rcousins)
			for clr in range(3):
				# same size
				_z = np.zeros_like(_rc[clr][0])
				_s = steerable.SteerablePyramid(_z, _z.shape[1], _z.shape[0], 1, num_ori, '', '', 0)

				_recon = np.zeros_like(_z)
				for k in range(num_ori):
					## modify angle: not good
#					amask = cr_mask(_s.AT[0], k, num_ori)
#					_recon = _recon + pyr_l[clr].BND[dp][k]['f'] * amask * _s.B_FILT[0][k]

					_recon = _recon + pyr_l[clr].BND[dp][k]['f'] * _s.B_FILT[0][k]
				
				_recon = _recon * _s.L0_FILT
				_recon = np.fft.ifft2(np.fft.ifftshift(_recon)).real

				# expand image created before and sum up
				_im = rec_im[clr]
				_im = sutils.expand(_im, 2).real / 4.
				_im = _im.real + _recon

				# adjust auto-correlation
				try:
					_im = sutils.mod_acorr(_im, orig_data.CF_CA[clr][dp], num_neighbor)
				except LinAlgError as e:
					LOGGER.info('Pass. LinAlgError {}'.format(e))

				# modify skewness
				_im = sutils.mod_skew(_im, orig_data.CF_MAR[clr][dp][2])

				# modify kurtosis
				_im = sutils.mod_kurt(_im, orig_data.CF_MAR[clr][dp][3])

				rec_im[clr] = _im

		# end of coarse to fine

		# ------------------------------------
		# Adjustment variance in H0 and final adjustment of coarse to fine.
		for clr in range(3):
			_tmp = pyr_l[clr].H0['s'].real
			_var = np.var(_tmp)
			_tmp = _tmp * np.sqrt(orig_data.H0_PRO[clr] / _var)

			# recon H0
			_recon = np.fft.fftshift(np.fft.fft2(_tmp))
			_recon = _recon * _s.H0_FILT
			_recon = np.fft.ifft2(np.fft.ifftshift(_recon)).real

			## this is final data of coarse to fine.
			rec_im[clr] = rec_im[clr] + _recon

			# adjust auto correlations
			rec_im[clr] = sutils.mod_acorr(rec_im[clr], orig_data.PCA_CA[clr], num_neighbor)

			# adjust skewness and kurtosis to original.
			_mean = np.mean(rec_im[clr])
			_var = np.var(rec_im[clr])
			rec_im[clr] = ( rec_im[clr] - _mean ) / np.sqrt(_var)
			## skewness
			rec_im[clr] = sutils.mod_skew(rec_im[clr], orig_data.MS_PCA[clr][2])
			## kurtosis
			rec_im[clr] = sutils.mod_kurt(rec_im[clr], orig_data.MS_PCA[clr][3])


		# ------------------------------------
		# Back to RBG channels and impose desired statistics
		_dim = im.shape
		im = im.reshape(_dim[0]*_dim[1], 3)
		for clr in range(3):
			im[:, clr] = rec_im[clr].flatten()

		_mean = np.mean(im, axis=0)

		im = im - _mean
		im = sutils.adjust_corr1(im, np.eye(3))
		## [Attn.] Bellow (1/4 power) may be mistake of textureColorAnalysis.m/textureColorSynthesis.m.
		## **(0.5) would be right. this obstructs color reproduction.
#		im = np.dot(im, np.dot(ocov_ediag**(0.25), ocov_evec.T))
		im = np.dot(im, np.dot(ocov_ediag**(0.5), ocov_evec.T))

		_mean = np.array([orig_data.MS_RGB[0][0], orig_data.MS_RGB[1][0], orig_data.MS_RGB[2][0]])
		im += _mean

		im = im.reshape(_dim[0], _dim[1], 3)

		# ------------------------------------
		# Adjust pixel statistic of RBG channels.
		for clr in range(3):
			# modify mean and variance of image created
			_mean = np.mean(im[:, :, clr])
			_var = np.var(im[:, :, clr])
			im[:, :, clr] = im[:, :, clr] - _mean
			im[:, :, clr] = im[:, :, clr] * np.sqrt(orig_data.RGB_MAR[clr][1] / _var)
			im[:, :, clr] = im[:, :, clr] + orig_data.MS_RGB[clr][0]
			# modify skewness of image created
			im[:, :, clr] = sutils.mod_skew(im[:, :, clr], orig_data.RGB_MAR[clr][2])
			# modify kurtsis of image created
			im[:, :, clr] = sutils.mod_kurt(im[:, :, clr], orig_data.RGB_MAR[clr][3])
			# adjust range
			_tmp = im[:, :, clr].reshape(_dim[0], _dim[1])
			_idx  = np.where(_tmp > orig_data.RGB_MAR[clr][4])
			_tmp[_idx] = orig_data.RGB_MAR[clr][4]
			im[:, :, clr] = _tmp
			_idx  = np.where(_tmp < orig_data.RGB_MAR[clr][5])
			_tmp[_idx] = orig_data.RGB_MAR[clr][5]
			im[:, :, clr] = _tmp

		# ------------------------------------
		# Save image
		# bugfix
		_o_img = Image.fromarray(np.uint8(im))
#		_o_img = Image.fromarray(np.uint8(im)).convert('L')
		_o_img.save(out_path + '/out-n{}-k{}-m{}-{}.png'.format(str(num_depth), str(num_ori), str(num_neighbor), str(it)))

		if it > 0:
			dst = np.sqrt(np.sum((prev_im - im)**2))
			rt = np.sqrt(np.sum((prev_im - im)**2)) / np.sqrt(np.sum(prev_im**2))
			LOGGER.debug('change {}, ratio {}'.format(str(dst), str(rt)))

			if it > 1:
				thr = np.abs(np.abs(prev_dst) - np.abs(dst)) / np.abs(prev_dst)
				LOGGER.debug('threshold {}'.format(str(thr)))
				if thr < 1e-6:
					break

			prev_dst = dst

		prev_im = im


'''
	make mask

	this is not good.

'''
#def cr_mask(angle, k, num_ori):
#	at = angle
#	th1, th2 = at, at
#
#	amask = np.zeros_like(at)
#	th1[np.where(at - k*np.pi/num_ori < -np.pi)] += 2.*np.pi
#	th1[np.where(at - k*np.pi/num_ori > np.pi)] -= 2.*np.pi
#	_ind = np.where(np.absolute(th1 - k*np.pi/num_ori) < np.pi/2.)
#	amask[_ind] = 2.
#	_ind = np.where(np.absolute(th1 - k*np.pi/num_ori) == np.pi/2.)
#	amask[_ind] = 1.
#	th2[np.where(at + (num_ori-k)*np.pi/4. < -np.pi)] += 2.*np.pi
#	th2[np.where(at + (num_ori-k)*np.pi/4. > np.pi)] -= 2.*np.pi
#	_ind = np.where(np.absolute(th2 + (num_ori-k) * np.pi/num_ori) < np.pi/2.)
#	amask[_ind] = 2.
#	_ind = np.where(np.absolute(th2 + (num_ori-k) * np.pi/num_ori) == np.pi/2.)
#	amask[_ind] = 1.
#
#	amask[int(amask.shape[0]/2), int(amask.shape[1]/2)] = 1.
#	amask[0, 0] = 1.
#	amask[0, amask.shape[1]-1] = 1.
#	amask[amask.shape[0]-1, 0] = 1.
#	amask[amask.shape[0]-1, amask.shape[1]-1] = 1.
#
#	return amask



if __name__ == "__main__":
	LOGGER.info('script start')
	
	start_time = time.time()

	parser = argparse.ArgumentParser(
	    description='Texture Synthesis (Color Version) by Portilla and Simoncelli')
	parser.add_argument('--orig_img', '-i', default='pebbles.jpg',
                    help='Original image')
	parser.add_argument('--out_dir', '-o', default='tmp',
                    help='Output directory')
	parser.add_argument('--num_depth', '-n', default=5, type=int,
                    help='depth of steerable pyramid')
	parser.add_argument('--num_ori', '-k', default=4, type=int,
                    help='orientation of steerable pyramid')
	parser.add_argument('--num_neighbor', '-m', default=7, type=int,
                    help='local neighborhood')
	parser.add_argument('--iter', default=100, type=int,
                    help='number of iterations')

	args = parser.parse_args()

	## validation of num. of neighbours.
	ms = [3, 5, 7, 9, 11, 13]
	if not args.num_neighbor in ms:
			LOGGER.error('illegal number of orientation: {}'.format(str(args.num_neighbor)))
			raise ValueError('illegal number of orientation: {}'.format(str(args.num_neighbor)))


	im = np.array(Image.open(args.orig_img))
	synthesis(im, im.shape[1], im.shape[0], args.num_depth, args.num_ori, args.num_neighbor, args.iter, args.out_dir)
