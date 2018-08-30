'''
	Texture Analysis (Color Version)

	this is a port of textureSynth/textureColorAnalysis.m by J. Portilla and E. Simoncelli.
	http://www.cns.nyu.edu/~lcv/texture/


'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from PIL import Image
import sys, os, copy
import logging

import sutils
import steerable_pyramid as steerable

SCRIPT_NAME = os.path.basename(__file__)
# logging
LOG_FMT = "[%(name)s] %(asctime)s %(levelname)s %(lineno)s %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FMT)
LOGGER = logging.getLogger(os.path.basename(__file__))

class TextureAnalysis():
	def __init__(self, image, xres, yres, n, k, m):
		self.IMAGE_ARRAY = image # array
		self.XRES = xres # horizontal resolution
		self.YRES = yres # vertical resolution
		self.PCA_ARRAY = np.array([])
		self.RGB_MAR = []
		self.K = k # num. of orientation
		self.N = n # depth
		self.M = m # window size (must be odd)

		### mean and covariances of original image
		self.MEAN_RGB = np.array([])
		self.COV_RGB = np.array([])

		### marginal statistics of original image
		self.MS_RGB = [] # RGB channels
		self.MS_PCA = [] # PCA channels

		### central suto correlation of pca channels
		self.PCA_CA = []

		## Steerable Pyramid
		self.SP = []
		self.LR = []
		self.LR_MAR = []
		self.LR_MMEAN = []
		self.LR_CA = []

		self.BND = []
		self.BND_M = []
		self.BND_MCOR = []
		self.BND_MMAR = []
		self.BND_R = []
		self.BND_P = []
		self.BND_RP = []
		self.BND_IP = []
		self.H0 = []
		self.H0_PRO = []

		self.COV_LR = np.array([])

		self.CF_MAR = []
		self.CF_CA = []
		self.CF_COUS = []
		self.CF_RCOU = []
		self.CF_CPAR = []
		self.CF_RPAR = []

	'''
		Analyse
	'''
	def analyse(self):

		# marginal statistics of RGB channels.
		for clr in range(3):
			self.RGB_MAR.append(sutils.mrg_stats(self.IMAGE_ARRAY[:, :, clr]))

		# means of orignal image
		self.MEAN_RGB = sutils.mean_im(self.IMAGE_ARRAY)
		# covariance matrix of orignal image
		self.COV_RGB = sutils.cov_im(self.IMAGE_ARRAY)

		# convert RGB to PCA
		self.PCA_ARRAY = sutils.get_pca(self.IMAGE_ARRAY)

		for clr in range(self.PCA_ARRAY.shape[2]):
			# marginal statistics of original image (R,G,B)
			_im = self.IMAGE_ARRAY[:,:,clr].reshape((self.IMAGE_ARRAY.shape[0], self.IMAGE_ARRAY.shape[1]))
			self.MS_RGB.append(sutils.mrg_stats(_im))

			#-----------------------------------------
			# principal components
			_pim = self.PCA_ARRAY[:,:,clr].reshape((self.PCA_ARRAY.shape[0], self.PCA_ARRAY.shape[1]))
			# marginal statistics of pca channels
			self.MS_PCA.append(sutils.mrg_stats(_pim))
			# (a1) central auto correlation of pca channels
			self.PCA_CA.append(sutils.get_acorr(_pim, self.M))

			#-----------------------------------------
			# create steerable pyramid
			_sp = steerable.SteerablePyramid(_pim, self.XRES, self.YRES, self.N, self.K, '', '', 0)
			_sp.create_pyramids()
			self.SP.append(copy.deepcopy(_sp))

			#-----------------------------------------
			# lowpass residual
			lr = copy.deepcopy(_sp.LR)
			## marginal statistics of LR
			self.LR_MMEAN.append(np.mean(np.abs(lr['s'])))
			## subtract mean : according to textureColorAnalysis.m
			_mean = np.mean(lr['s'].real)
			lr['s'] = lr['s'].real - _mean
			lr['f'] = np.fft.fftshift(np.fft.fft2(lr['s']))
			self.LR.append(lr)
			## marginal statistics of lowpass residual
			## get L0 of LR of small size.(this tric is for synthesis process)
			_s = steerable.SteerablePyramid(lr['s'], lr['s'].shape[1], lr['s'].shape[0], 1, 4, '', '', 0)
			_s.create_pyramids()

			# initial value of coarse to fine
			im = _s.L0['s'].real
			## marginal statistics of LR
			self.LR_MAR.append(sutils.mrg_stats(im))
			## central auto correlation of lowpass residuals
			self.LR_CA.append(sutils.get_acorr(im, self.M))

			#-----------------------------------------
			# bandpass
			bnd = copy.deepcopy(_sp.BND)
			self.BND.append(bnd)

			_b_m, _b_r, _b_i = sutils.trans_b(copy.deepcopy(_sp.BND))
			## marginal statistics of magnitude
			self.BND_MMAR.append(sutils.mrg_b(_b_m))
			## magnitude
			for i in range(len(_b_m)):
				for k in range(len(_b_m[i])):
					_b_m[i][k] -= np.mean(_b_m[i][k])
			self.BND_M.append(_b_m)
			## central auto-correlation of magnitude (this is 'ace' in textureColorAnalysis.m)
			self.BND_MCOR.append(sutils.autocorr_b(_b_m, self.M))
			## real values
			self.BND_R.append(_b_r)

			_b_p, _b_rp, _b_ip = sutils.get_parent(copy.deepcopy(_sp.BND), lr)
			## maginitude of parent bandpass  (this is 'parent' in textureColorAnalysis.m)
			self.BND_P.append(_b_p)
			## real values of parent bandpass (this is half of 'rparent' in textureColorAnalysis.m)
			self.BND_RP.append(_b_rp)
			## imaginary values of parent bandpass (this is half of 'rparent' in textureColorAnalysis.m)
			self.BND_IP.append(_b_ip)

			#-----------------------------------------
			# highpass residual
			_b = copy.deepcopy(_sp.H0)
			self.H0.append(_b)
			## marginal statistics of highpass residual
			self.H0_PRO.append(np.var(_b['s'].real))

			#-----------------------------------------
			# statistics for coarse to fine

			# coarse to fine loop
			_ms = []
			_ac = []
			_cou = []
			for dp in range(self.N-1, -1, -1):
				# create steerable pyramid (create filters only)
				_z = np.zeros_like(bnd[dp][0]['s'])
				_s = steerable.SteerablePyramid(_z, _z.shape[1], _z.shape[0], 1, self.K, '', '', 0)
				# reconstruct dummy pyramid
				_recon = np.zeros_like(_z)
				for k in range(self.K):
					_recon += _s.B_FILT[0][k] * bnd[dp][k]['f']
				_recon = _recon * _s.L0_FILT
				_recon = np.fft.ifft2(np.fft.ifftshift(_recon)).real

				# expand
				im = sutils.expand(im, 2).real / 4.
				im = im.real + _recon

				# marginal statistics
				_ms.append(sutils.mrg_stats(im))			
				# central auto correlations
				_ac.append(sutils.get_acorr(im, self.M))	

			self.CF_MAR.append(_ms[::-1])
			self.CF_CA.append(_ac[::-1]) # this is 'acr' in textureColorAnalysis.m

		#-----------------------------------------
		# auto correlartion matrix of lowpass residual (2 slided)
		self.COV_LR = sutils.cov_lr(self.LR)


		# coarse to fine loop (Get statistics of Bandpass)
		for dp in range(self.N-1, -1, -1):
				
			# combine colors
			cousins = sutils.cclr_b(self.BND_M, dp)
			## save covariance matrices
			_tmp = np.dot(cousins.T, cousins) / cousins.shape[0]
			self.CF_COUS.append(copy.deepcopy(_tmp))

			bnd_r = []
			for clr in range(3):
				_list = []
				for k in range(self.K):
					_list.append(self.BND[clr][dp][k]['s'].real)
				bnd_r.append(_list)
			
			rcousins = sutils.cclr_bc(bnd_r, dp)
			# save covariance matrices
			_tmp = np.dot(rcousins.T, rcousins) / rcousins.shape[0]
			self.CF_RCOU.append(copy.deepcopy(_tmp))
			
			rparents = sutils.cclr_rp(self.BND_RP, self.BND_IP, dp)
			# save covariance matrices
			_tmp = np.dot(rcousins.T, rparents) / rcousins.shape[0]
			self.CF_RPAR.append(copy.deepcopy(_tmp))

			if dp < self.N-1:
				parents = sutils.cclr_p(self.BND_P, dp)
				# save covariance matrices
				_tmp = np.dot(cousins.T, parents) / cousins.shape[0]
				self.CF_CPAR.append(copy.deepcopy(_tmp))


		self.CF_COUS = self.CF_COUS[::-1]
		self.CF_RCOU = self.CF_RCOU[::-1]
		self.CF_RPAR = self.CF_RPAR[::-1]
		self.CF_CPAR = self.CF_CPAR[::-1]

		return None


