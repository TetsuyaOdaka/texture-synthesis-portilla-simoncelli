'''
	Texture Synthesis (Gray Scale Version)

	this is a port of textureSynth/textureSynthesis.m by J. Portilla and E. Simoncelli.
	http://www.cns.nyu.edu/~lcv/texture/

	Differences:
	(1) i use real version of steerable pyramid.
	    Sorry. I could not understand the algorithm of complex version in textureSynthesis.m.
	(2) i don't use filter masks of orientations in the process of coarse to fine.

	Usage:
	python texture_synthesis_g.py -i radish-mono.jpg -o tmp -n 5 -k 4 -m 7 --iter 100

	-i : input image
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
import texture_analysis_g as ta

SCRIPT_NAME = os.path.basename(__file__)

# logging
LOG_FMT = "[%(name)s] %(asctime)s %(levelname)s %(lineno)s %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FMT)
LOGGER = logging.getLogger(os.path.basename(__file__))

ALPHA = 0.8

'''
	Texture Synthesis by Portilla-Simoncelli's algorithm

'''
def synthesis(image, resol_x, resol_y, num_depth, num_ori, num_neighbor, iter, out_path):
	# analyse original image
	orig_data = ta.TextureAnalysis(image, resol_x, resol_y, num_depth, num_ori, num_neighbor)
	orig_data.analyse()

	# initialize random image
	im = np.random.normal(0, 1, resol_x * resol_y).reshape(resol_y, resol_x)
	im = im * np.sqrt(orig_data.IM_VAR)
	im = im + orig_data.IM_MAR[0]

	# iteration
	prev_im = np.array([])
	prev_dst = 0.

	for it in range(0, iter):
		LOGGER.debug('iteration {}'.format(str(it)))
	
		pyr_l = []
		lr_l = []

		# ------------------------------------
		# Create pyramids of each PCA channel
		# steerable pyramid
		_sp = steerable.SteerablePyramid(im, resol_x, resol_y, num_depth, num_ori, '', '', 0)
		_sp.create_pyramids()

		# subtract means from lowpass residuals
		_sp.LR['s'] = _sp.LR['s'].real - np.mean(_sp.LR['s'].real.flatten())

		pyr_l = copy.deepcopy(_sp)
		lr_l = _sp.LR

		# ------------------------------------
		# Adjust lowpass residual and get initial image for coarse to fine
		# modify central auto correlation
		try:
			lr_l['s'] = sutils.mod_acorr(lr_l['s'], orig_data.LR_CA, num_neighbor)
		except LinAlgError as e:
			LOGGER.info('LinAlgError {}'.format(e))
			lr_l['s'] = lr_l['s'] * np.sqrt(orig_data.LR_MAR[1] / np.var(lr_l['s']))

		lr_l['s'] = lr_l['s'].real
		# modify skewness of lowpass residual
		try:
			lr_l['s'] = sutils.mod_skew(lr_l['s'], orig_data.LR_MAR[2])
		except LinAlgError as e:
			LOGGER.info('LinAlgError {}'.format(e))
		# modify kurtosis of lowpass residual
		try:
			lr_l['s'] = sutils.mod_kurt(lr_l['s'], orig_data.LR_MAR[3])
		except LinAlgError as e:
			LOGGER.info('LinAlgError {}'.format(e))

		lr_l['f'] = np.fft.fftshift(np.fft.fft2(lr_l['s']))

		 # initial coarse to fine
		rec_im = lr_l['s']

		## get original statistics of bandpass signals.
		# create parents
		bnd = copy.deepcopy(pyr_l.BND)
		_b_m, _, _ = sutils.trans_b(pyr_l.BND)
		for i in range(len(_b_m)):
			for k in range(len(_b_m[i])):
				_b_m[i][k] -= np.mean(_b_m[i][k])
		## magnitude
		bnd_m = _b_m

		_b_p, _b_rp, _b_ip = sutils.get_parent_g(pyr_l.BND, pyr_l.LR)
		## maginitude of parent bandpass  (this is 'parent' in textureColorAnalysis.m)
		bnd_p = _b_p
		## real values of parent bandpass (this is half of 'rparent' in textureColorAnalysis.m)
		bnd_rp = _b_rp
		## imaginary values of parent bandpass (this is half of 'rparent' in textureColorAnalysis.m)
		bnd_ip = _b_ip

		# ------------------------------------
		# Coarse to fine adjustment
		for dp in range(num_depth-1, -1, -1):

			# combine orientations
			cousins = sutils.cori_b(bnd_m, dp)

			# adjust covariances
			_prev = cousins
			if dp < num_depth-1:
				parents = bnd_p[dp]
				cousins = sutils.adjust_corr2(_prev, orig_data.CF_COUS[dp], parents, orig_data.CF_CPAR[dp])
				if np.isnan(cousins).any():
					LOGGER.info('NaN in adjust_corr2')
					cousins = sutils.adjust_corr1(_prev, orig_data.CF_COUS[dp])
				rparents = sutils.cori_rp(bnd_rp, bnd_ip, dp)
			else:
				cousins = sutils.adjust_corr1(_prev, orig_data.CF_COUS[dp])

			# separate orientations
			cousins = sutils.sori_b(cousins, num_ori)

			# adjust central auto corr. and update bandpass.
			bnd_r = []
			for k in range(num_ori):
				# adjust central auto-correlations
				try:
					_tmp = sutils.mod_acorr(cousins[k], orig_data.BND_MCOR[dp][k], num_neighbor)
				except LinAlgError as e:
					LOGGER.info('LinAlgError {}'.format(e))
					_tmp = cousins[k]

				# update BND_N
				bnd_m[dp][k] = _tmp
				_mean = orig_data.BND_MMAR[dp][k][0]
				_tmp = _tmp + _mean
				_idx = np.where(_tmp < 0)
				_tmp[_idx] = 0

				_bnd = pyr_l.BND[dp][k]['s']
				_idx1 = np.where(np.abs(_bnd) < 10**(-12))
				_idx2 = np.where(np.abs(_bnd) >= 10**(-12))
				_bnd[_idx1] = _bnd[_idx1] * _tmp[_idx1]
				_bnd[_idx2] = _bnd[_idx2] * _tmp[_idx2] / np.abs(_bnd[_idx2])

				bnd_r.append(_bnd.real)

			# combine orientations & make rcousins
			rcousins = sutils.cori_bc(bnd_r, dp)

			# adjust cross-correlation of real values of B and real/imaginary values of parents
			_prev = rcousins
			try:
				if dp < num_depth-1:
					rcousins = sutils.adjust_corr2(_prev, orig_data.CF_RCOU[dp], rparents, orig_data.CF_RPAR[dp])
					if np.isnan(rcousins).any():
						LOGGER.info('NaN in adjust_corr2')
						rcousins = sutils.adjust_corr1(_prev, orig_data.CF_RCOU[dp])
						if np.isnan(rcousins).any():
							LOGGER.info('NaN in adjust_corr1')
							rcousins = _prev
				else:
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

			# separate orientations
			rcousins = sutils.sori_b(rcousins, num_ori)
			for k in range(num_ori):
				## update pyramid
				pyr_l.BND[dp][k]['s'] = rcousins[k]
				pyr_l.BND[dp][k]['f'] = np.fft.fftshift(np.fft.fft2(rcousins[k]))

			# combine bands
			_rc = copy.deepcopy(rcousins)
			# same size
			_z = np.zeros_like(_rc[0])
			_s = steerable.SteerablePyramid(_z, _z.shape[1], _z.shape[0], 1, num_ori, '', '', 0)

			_recon = np.zeros_like(_z)
			for k in range(num_ori):
				_recon = _recon + pyr_l.BND[dp][k]['f'] * _s.B_FILT[0][k]
			_recon = _recon * _s.L0_FILT
			_recon = np.fft.ifft2(np.fft.ifftshift(_recon)).real

			# expand image created before and sum up
			_im = rec_im
			_im = sutils.expand(_im, 2).real / 4.
			_im = _im.real + _recon

			# adjust auto-correlation
			try:
				_im = sutils.mod_acorr(_im, orig_data.CF_CA[dp], num_neighbor)
			except LinAlgError as e:
				LOGGER.info('Pass. LinAlgError {}'.format(e))

			# modify skewness
			try:
				_im = sutils.mod_skew(_im, orig_data.CF_MAR[dp][2])
			except LinAlgError as e:
				LOGGER.info('LinAlgError {}'.format(e))

			# modify kurtosis
			try:
				_im = sutils.mod_kurt(_im, orig_data.CF_MAR[dp][3])
			except LinAlgError as e:
				LOGGER.info('LinAlgError {}'.format(e))

			rec_im = _im

		# end of coarse to fine

		# ------------------------------------
		# Adjustment variance in H0 and final adjustment of coarse to fine.
		_tmp = pyr_l.H0['s'].real
		_var = np.var(_tmp)
		_tmp = _tmp * np.sqrt(orig_data.H0_PRO / _var)

		# recon H0
		_recon = np.fft.fftshift(np.fft.fft2(_tmp))
		_recon = _recon * _s.H0_FILT
		_recon = np.fft.ifft2(np.fft.ifftshift(_recon)).real

		## this is final data of coarse to fine.
		rec_im = rec_im + _recon

		# adjust skewness and kurtosis to original.
		_mean = np.mean(rec_im)
		_var = np.var(rec_im)
		rec_im = ( rec_im - _mean ) * np.sqrt( orig_data.IM_MAR[1] / _var)
		rec_im = rec_im + orig_data.IM_MAR[0]

		## skewness
		rec_im = sutils.mod_skew(rec_im, orig_data.IM_MAR[2])
		## kurtosis
		rec_im = sutils.mod_kurt(rec_im, orig_data.IM_MAR[3])

		_idx  = np.where(rec_im > orig_data.IM_MAR[4])
		rec_im[_idx] = orig_data.IM_MAR[4]
		_idx  = np.where(rec_im < orig_data.IM_MAR[5])
		rec_im[_idx] = orig_data.IM_MAR[5]

		im = rec_im

		# ------------------------------------
		# Save image
		_o_img = Image.fromarray(np.uint8(im)).convert('L')
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

			# acceleration
			im = im + ALPHA * (im - prev_im)

		prev_im = im



if __name__ == "__main__":
	LOGGER.info('script start')
	
	start_time = time.time()

	parser = argparse.ArgumentParser(
	    description='Texture Synthesis (Gray Version) by Portilla and Simoncelli')
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
