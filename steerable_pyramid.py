'''
	Steerable Pyramid (Comlex Version)

	This is just for TextureSynthesis.
	You can not reconstruct the images from the results of decomposition

	==================================================================================

	This implementaion is basically based on J. Portilla and E. Simoncelli [2000] .
	The definition on bandpass filters are based on T. Briand et al. [2014].
	Spatioal representation is in complex. 

	In this program, all filters are applied in Fourier domain because of computational efficiency.
	As described in . Briand et al. [2014] IPOL

	"Parametric Texture Model Based on Joint Statistics of Complex Wavelet Coefficient"
	J. Portilla and E. Simoncelli [2000]
	http://www.cns.nyu.edu/pub/lcv/portilla99.pdf

	According to MatLab source code distributed by E.Simoncelli, filters are applied in
	spatioal domain.
	https://github.com/gregfreeman/matlabPyrTools

	See also,
	"The Heeger-Bergen Pyramid-Based Texture Synthesis Algorithm"
	T. Briand et al. [2014] IPOL
	http://www.ipol.im/pub/art/2014/79/

	"The Steerable Pyramid:A Flexible Architecture For Multi-Scale Derivative Computation"
	E.Simoncelli and W.Freeman [1995]
	http://www.cns.nyu.edu/pub/eero/simoncelli95b.pdf

	"A Filter Design Technique For Steerable Pyramid Image Transform"
	A.Karasaridis and E.Simoncelli [1996]
	https://pdfs.semanticscholar.org/625e/ec8262570a3d62a2f252c151ef14e2be9b5d.pdf

	"Design and Use of Steerable Filters"
	W.Freeman and E.Adelson [1991]
	http://people.csail.mit.edu/billf/publications/Design_and_Use_of_Steerable_Filters.pdf


'''

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import sys, os
import logging

SCRIPT_NAME = os.path.basename(__file__)
# logging
LOG_FMT = "[%(name)s] %(asctime)s %(levelname)s %(lineno)s %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FMT)
LOGGER = logging.getLogger(os.path.basename(__file__))

'''

	Steerable Pyramid

'''

class SteerablePyramid():
	def __init__(self, image, xres, yres, n, k, image_name, out_path, verbose):
		self.XRES = xres # horizontal resolution
		self.YRES = yres # vertical resolution
		self.IMAGE_ARRAY = np.asarray(image, dtype='complex')
		self.IMAGE_NAME = image_name
#		self.OUT_PATH = out_path # path to the directory for saving images.
		self.OUT_PATH = out_path + '/{}' # path to the directory for saving images.
		## validation of num. of orientaion
		self.Ks = [4, 6, 8, 10, 12, 15, 18, 20, 30, 60]
		if not k in self.Ks:
			LOGGER.error('illegal number of orientation: {}'.format(str(k)))
			raise ValueError('illegal number of orientation: {}'.format(str(k)))
		self.K = k # num. of orientation
		## validation of depth
		_tmp = np.log2(np.min(np.array([xres, yres])))
		if n  > _tmp - 1:
			LOGGER.error('illegal depth: {}'.format(str(n)))
			raise ValueError('illegal depth: {}'.format(str(n)))
		self.N = n # depth
		self.verbose = verbose # verbose
		self.ALPHAK = 2.**(self.K-1) * math.factorial(self.K-1)/np.sqrt(self.K * float(math.factorial(2.*(self.K-1))))
		self.RES = []
		for i in range(0, self.N):
			_tmp = 2.** i
			self.RES.append( (int(self.XRES/_tmp), int(self.YRES/_tmp)) )
		self.GRID = [] # grid
		self.WX = []
		self.WY = []
		for i in range(0, self.N):
			_x = np.linspace(-np.pi, np.pi, num = self.RES[i][0], endpoint = False)
			_y = np.linspace(-np.pi, np.pi, num = self.RES[i][1], endpoint = False)
			self.WX.append(_x)
			self.WY.append(_y)
			self.GRID.append(np.zeros((_x.shape[0], _y.shape[0])))

		self.RS = [] # polar coordinates
		self.AT = [] # angular cordinates

		# Filters
		self.H0_FILT = np.array([])
		self.L0_FILT = np.array([])
		self.L_FILT = []
		self.H_FILT = []
		self.B_FILT = []

		# Pyramids
		self.H0 = {'f':None, 's':None}
		self.L0 = {'f':None, 's':None}
		self.LR = {'f':None, 's':None}
		self.BND = []
		self.LOW = [] # L1, ...LN

		## CREATE FILTERS
		# caliculate polar coordinates.
		self.RS, self.AT = self.caliculate_polar()
		## for debugging, let coordinates same as Matlab.
#		for n in range(self.N):
#			self.RS[n] = self.RS[n].T
#			self.AT[n] = self.AT[n].T

		# caliculate H0 values on the grid.
		fil = self.calicurate_h0_filter()
		self.H0_FILT = fil

		# caliculate L0 values on the grid.
		fil = self.calicurate_l0_filter()
		self.L0_FILT = fil

		# caliculate L(Low pass filter) values on the grid. 
		fil = self.calicurate_l_filter()
		self.L_FILT = fil

		# caliculate H(fot bandpass filter) values on the grid.
		fil = self.calicurate_h_filter()
		self.H_FILT = fil

		# caliculate B values on the grid.
		fils = self.calicurate_b_filters()
		self.B_FILT = fils

	# caliculate polar coordinates on the grid.
	def caliculate_polar(self):
		pol = []
		ang = []
		for i in range(0, self.N):
			# caliculate polar coordinates(radius) on the grid. they are in [0, inf).
			rs = self.GRID[i].copy()
			yy, xx= np.meshgrid(self.WX[i], self.WY[i])
			rs = np.sqrt((xx)**2 + (yy)**2)

			# caliculate angular coordinates(theta) on the grid. they are in (-pi, pi].
			at= self.GRID[i].copy()
			_idx = np.where((yy == 0) & (xx < 0))
			at[_idx] = np.pi
			_idx = np.where((yy != 0) | (xx >= 0))
			at[_idx] = np.arctan2(yy[_idx], xx[_idx])

			pol.append(rs)
			ang.append(at)

		return pol, ang

	# caliculate H0 values on the grid.
	def calicurate_h0_filter(self):
		fil = self.GRID[0].copy()
		fil[np.where(self.RS[0] >= np.pi)] = 1
		fil[np.where(self.RS[0] < np.pi/2.)] = 0
		_ind = np.where((self.RS[0] > np.pi/2.) & (self.RS[0] < np.pi))
		fil[_ind] = np.cos(np.pi/2. * np.log2( self.RS[0][_ind]/np.pi) )

		if self.verbose == 1:
			# save image
			plt.clf()
			plt.contourf(self.WX[0], self.WY[0], fil)
			plt.axes().set_aspect('equal', 'datalim')
			plt.colorbar()
			plt.xlabel('x')
			plt.ylabel('y')
			plt.title('H0 Filter : Fourier Domain')
			plt.savefig(self.OUT_PATH.format('fil_highpass0.png'))

		return fil

	# caliculate L0 values on the grid.
	def calicurate_l0_filter(self):
		fil = self.GRID[0].copy()
		fil[np.where(self.RS[0] >= np.pi)] = 0
		fil[np.where(self.RS[0] <= np.pi/2.)] = 1
		_ind = np.where((self.RS[0] > np.pi/2.) & (self.RS[0] < np.pi))
		fil[_ind] = np.cos(np.pi/2. * np.log2(2. * self.RS[0][_ind]/np.pi))

		if self.verbose == 1:
			# save image
			plt.clf()
			plt.contourf(self.WX[0], self.WY[0], fil)
			plt.axes().set_aspect('equal', 'datalim')
			plt.colorbar()
			plt.xlabel('x')
			plt.ylabel('y')
			plt.title('L0 Filter : Fourier Domain')
			plt.savefig(self.OUT_PATH.format('fil_lowpass0.png'))

		return fil

	# caliculate L filter values on the grid.
	def calicurate_l_filter(self):
		_f = []
		for i in range(0, self.N):
			fil = self.GRID[i].copy()
			fil[np.where(self.RS[i] >= np.pi/2.)] = 0
			fil[np.where(self.RS[i] <= np.pi/4.)] = 1
			_ind = np.where((self.RS[i] > np.pi/4.) & (self.RS[i] < np.pi/2.))
			fil[_ind] = np.cos(np.pi/2. * np.log2(4. * self.RS[i][_ind]/np.pi))

			_f.append(fil)

			if i == 0 and self.verbose == 1:
				plt.clf()
				plt.contourf(self.WX[i], self.WY[i], fil)
				plt.axes().set_aspect('equal', 'datalim')
				plt.colorbar()
				plt.xlabel('x')
				plt.ylabel('y')
				plt.title('Lowpass filter of Layer{} : Fourier Domain'.format(str(i)))
				plt.savefig(self.OUT_PATH.format('fil_lowpass-layer{}.png'.format(str(i))))

		return _f

	# caliculate H0 filter values on the grid.
	def calicurate_h_filter(self):
		_f = []
		for i in range(0, self.N):
			fil = self.GRID[i].copy()
			fil[np.where(self.RS[i] >= np.pi/2.)] = 1
			fil[np.where(self.RS[i] <= np.pi/4.)] = 0
			_ind = np.where((self.RS[i] > np.pi/4.) & (self.RS[i] < np.pi/2.))
			fil[_ind] = np.cos(np.pi/2. * np.log2(2.*self.RS[i][_ind]/np.pi))

			_f.append(fil)		

			if i == 0 and self.verbose == 1:
				plt.clf()
				plt.contourf(self.WX[i], self.WY[i], fil)
				plt.axes().set_aspect('equal', 'datalim')
				plt.colorbar()
				plt.xlabel('x')
				plt.ylabel('y')
				plt.title('Highpass filter of Layer{} : Fourier Domain'.format(str(i)))
				plt.savefig(self.OUT_PATH.format('fil_highpass-layer{}.png'.format(str(i))))

		return _f

	def calicurate_b_filters(self):
		f_ = []
		for i in range(0, self.N):
			fils_ = []

			for k in range(self.K):
				# caliculate Bk values on the grid.
				fil_= np.zeros_like(self.GRID[i], dtype=complex)
				th1= self.AT[i].copy()
				th2= self.AT[i].copy()

				th1[np.where(self.AT[i] - k*np.pi/self.K < -np.pi)] += 2.*np.pi
				th1[np.where(self.AT[i] - k*np.pi/self.K > np.pi)] -= 2.*np.pi
				ind_ = np.where(np.absolute(th1 - k*np.pi/self.K) <= np.pi/2.)
				fil_[ind_] = self.ALPHAK * (np.cos(th1[ind_] - k*np.pi/self.K))**(self.K-1)
#				fil_[ind_] = complex(0,1)**k * self.ALPHAK * (np.cos(th1[ind_] - k*np.pi/self.K))**(self.K-1)
				th2[np.where(self.AT[i] + (self.K-k)*np.pi/self.K < -np.pi)] += 2.*np.pi
				th2[np.where(self.AT[i] + (self.K-k)*np.pi/self.K > np.pi)] -= 2.*np.pi
				ind_ = np.where(np.absolute(th2 + (self.K-k) * np.pi/self.K) <= np.pi/2.)
				fil_[ind_] = self.ALPHAK * (np.cos(th2[ind_]+ (self.K-k) * np.pi/self.K))**(self.K-1)
#				fil_[ind_] = complex(0,1)**k * self.ALPHAK * (np.cos(th2[ind_]+ (self.K-k) * np.pi/self.K))**(self.K-1)

				fil_= self.H_FILT[i] * fil_
				fils_.append(fil_.copy())

				if i == 0 and self.verbose == 1:
					plt.clf()
					plt.contourf(self.WX[i], self.WY[i], np.abs(fil_))
					plt.axes().set_aspect('equal', 'datalim')
					plt.colorbar()
					plt.xlabel('x')
					plt.ylabel('y')
					plt.title('Bandpass filter of layer{} : Fourier Domain'.format(str(i)))
					plt.savefig(self.OUT_PATH.format('fil_bandpass{}-layer{}.png'.format(str(k), str(i))))

					plt.clf()
					plt.contourf(self.WX[i], self.WY[i], np.abs(fil_ * self.L0_FILT))
					plt.axes().set_aspect('equal', 'datalim')
					plt.colorbar()
					plt.xlabel('x')
					plt.ylabel('y')
					plt.title('Bandpass * Lowpass filter of layer{}'.format(str(i)))
					plt.savefig(self.OUT_PATH.format('fil_lo-bandpass{}-layer{}.png'.format(str(k), str(i))))
	
			f_.append(fils_)

		return f_

	# create steerable pyramid
	def create_pyramids(self):

		# DFT
		ft = np.fft.fft2(self.IMAGE_ARRAY)
		_ft = np.fft.fftshift(ft)

		# apply highpass filter(H0) and save highpass resudual
		h0 = _ft * self.H0_FILT
		f_ishift = np.fft.ifftshift(h0)
		img_back = np.fft.ifft2(f_ishift)
		# frequency
		self.H0['f'] = h0.copy()
		# space
		self.H0['s'] = img_back.copy()

		if self.verbose == 1:
			_tmp = np.absolute(img_back)
			Image.fromarray(np.uint8(_tmp), mode='L').save(self.OUT_PATH.format('{}-h0.png'.format(self.IMAGE_NAME)))

		# apply lowpass filter(L0).
		l0 = _ft * self.L0_FILT
		f_ishift = np.fft.ifftshift(l0)
		img_back = np.fft.ifft2(f_ishift)
		self.L0['f'] = l0.copy()
		self.L0['s'] = img_back.copy()

		if self.verbose == 1:
			_tmp = np.absolute(img_back)
			Image.fromarray(np.uint8(_tmp), mode='L').save(self.OUT_PATH.format('{}-l0.png'.format(self.IMAGE_NAME)))

		# apply bandpass filter(B) and downsample iteratively. save pyramid
		_last = l0
		for i in range(self.N):
			_t = []
			for j in range(len(self.B_FILT[i])):
				_tmp = {'f':None, 's':None}
				lb = _last * self.B_FILT[i][j]
				f_ishift = np.fft.ifftshift(lb)
				img_back = np.fft.ifft2(f_ishift)
				# frequency
				_tmp['f'] = lb
				# space
				_tmp['s'] = img_back
				_t.append(_tmp)

				if self.verbose == 1:
					_tmp = np.absolute(img_back.real)
					Image.fromarray(np.uint8(_tmp), mode='L').save(self.OUT_PATH.format('{}-layer{}-lb{}.png'.format(self.IMAGE_NAME, str(i), str(j))))
		
			self.BND.append(_t.copy())

			# apply lowpass filter(L) to image(Fourier Domain) downsampled.
			l1 = _last * self.L_FILT[i]

			## Downsampling
			# filter for cutting off high frequerncy(>np.pi/2).
			# (Attn) steerable pyramid is basically anti-aliases. see http://www.cns.nyu.edu/pub/eero/simoncelli95b.pdf
			# this filter is not needed actually ,but prove anti-aliases characteristic of the steerable filters.
			down_fil = np.zeros(_last.shape)
			quant4x = int(down_fil.shape[1]/4)
			quant4y = int(down_fil.shape[0]/4)
			down_fil[quant4y:3*quant4y, quant4x:3*quant4x] = 1

			# apply downsample filter.
			dl1 = l1 * down_fil

			# extract the central part of DFT
			down_image = np.zeros((2*quant4y, 2*quant4x), dtype=complex)
			down_image = dl1[quant4y:3*quant4y, quant4x:3*quant4x]
#
			f_ishift = np.fft.ifftshift(down_image)
			img_back = np.fft.ifft2(f_ishift)
			self.LOW.append({'f':down_image, 's':img_back})
			
			if self.verbose == 1:
				_tmp = np.absolute(img_back)
				Image.fromarray(np.uint8(_tmp), mode='L').save(self.OUT_PATH.format('{}-residual-layer{}.png'.format(self.IMAGE_NAME, str(i))))

			_last = down_image

		# lowpass residual
		self.LR['f'] = _last.copy()
		self.LR['s'] = img_back.copy()

		return None


	# image reconstruction from steerable pyramid in Fourier domain.
	def collapse_pyramids(self):
		_resid = self.LR['f']

		for i in range(self.N-1,-1,-1):
			## upsample residual
			_tmp_tup = tuple(int(2*x) for x in _resid.shape)
			_tmp = np.zeros(_tmp_tup, dtype=np.complex)
			quant4x = int(_resid.shape[1]/2)
			quant4y = int(_resid.shape[0]/2)
			_tmp[quant4y:3*quant4y, quant4x:3*quant4x] = _resid
			_resid = _tmp

			_resid = _resid * self.L_FILT[i]
			for j in range(len(self.B_FILT[i])):
				_resid += self.BND[i][j]['f'] * self.B_FILT[i][j]
	
		# finally reconstruction is done.
		recon = _resid * self.L0_FILT + self.H0['f'] * self.H0_FILT

		return recon

	# clear the steerable pyramid
	def clear_pyramids(self):
		self.H0['f'] = np.zeros_like(self.H0['f'])
		self.H0['s'] = np.zeros_like(self.H0['s'])
		self.L0['f'] = np.zeros_like(self.L0['f'])
		self.L0['s'] = np.zeros_like(self.L0['s'])
		self.LR['f'] = np.zeros_like(self.LR['f'])
		self.LR['s'] = np.zeros_like(self.LR['s'])

		for i in range(len(self.BND)):
			for j in range(len(self.BND[i])):
				self.BND[i][j]['s'] = np.zeros_like(self.BND[i][j]['s'])
				self.BND[i][j]['f'] = np.zeros_like(self.BND[i][j]['f'])
		
		for i in range(len(self.LOW)):
			self.LOW[i]['s'] = np.zeros_like(self.LOW[i]['s'])
			self.LOW[i]['f'] = np.zeros_like(self.LOW[i]['f'])

		return


