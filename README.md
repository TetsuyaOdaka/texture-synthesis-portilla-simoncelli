# Texture Sysnthesis by Portila and Simoncelli

Python3 implementation of Polltila and Simoncelli(2000)'s texture synthesis.  
  
[Parametric Texture Model Based on Joint Statistics of Complex Wavelet Coefficient, Portilla, J. and Simoncelli, E.(2000) ](http://www.cns.nyu.edu/pub/lcv/portilla99.pdf)  
  

<br/>Basically this repository is a port of [Matlab programs of NYU](https://github.com/LabForComputationalVision/textureSynth).But "steerable pyramid" is based on Briand et al.(2014) .  

[The Heeger-Bergen Pyramid-Based Texture Synthesis Algorithm, Briand,T. et al. (2014)](http://www.ipol.im/pub/art/2014/79/)
  
  
<br/>I have already published the [python3 implementation of steerable pyramid](https://github.com/TetsuyaOdaka/SteerablePyramid/) by their equations.  


## Results
*Attentions !!*  
All images(except results) are from [NYU websites](http://www.cns.nyu.edu/~lcv/texture/). The copyrights belongs to it.


### Gray Scale
#### Original image
<img src="https://github.com/TetsuyaOdaka/texture-synthesis-portilla-simoncelli/blob/master/samples/bark.jpg" alt="texture synthesis">  

#### Synthesized image
<img src="https://github.com/TetsuyaOdaka/texture-synthesis-portilla-simoncelli/blob/master/samples/bark-out.png" alt="texture synthesisd">  
  
#### Original image
<img src="https://github.com/TetsuyaOdaka/texture-synthesis-portilla-simoncelli/blob/master/samples/jrotpluses.jpg" alt="texture synthesis">  

#### Synthesized image
<img src="https://github.com/TetsuyaOdaka/texture-synthesis-portilla-simoncelli/blob/master/samples/jrotpluses-out.png" alt="texture synthesisd">  


### Color Version
#### Original image
<img src="https://github.com/TetsuyaOdaka/texture-synthesis-portilla-simoncelli/blob/master/samples/radish.jpg" alt="texture synthesis">  

#### Synthesized image
<img src="https://github.com/TetsuyaOdaka/texture-synthesis-portilla-simoncelli/blob/master/samples/radish-out.png" alt="texture synthesis">  

#### Original image
<img src="https://github.com/TetsuyaOdaka/texture-synthesis-portilla-simoncelli/blob/master/samples/pebbles.jpg" alt="texture synthesis">  

#### Synthesized image
<img src="https://github.com/TetsuyaOdaka/texture-synthesis-portilla-simoncelli/blob/master/samples/pebbles-out.png" alt="texture synthesis">  


## Usage 
### Environment
- python3.5 (3.0+)
- GPU is not used.  

### Execution(Gray Scale Version)
- create 'out' directory. 
- `python texture_synthesis_g.py -i radish-mono.jpg -o out -n 5 -k 4 -m 7 --iter 100`,  if you want to know details about parameters, see [source code](https://github.com/TetsuyaOdaka/texture-synthesis-portilla-simoncelli/blob/master/texture_analysis_g.py)  


### Execution(RGB Color Version)
- create 'out' directory. 
- `python texture_synthesis_g.py -i radish.jpg -o out -n 5 -k 4 -m 7 --iter 100`,  if you want to know details about parameters, see [source code](https://github.com/TetsuyaOdaka/texture-synthesis-portilla-simoncelli/blob/master/texture_analysis.py)  


## Acknowledgement
thanks to the authors of NYU.
- [Matlab programs of NYU](https://github.com/LabForComputationalVision/textureSynth)


## References
- [Parametric Texture Model Based on Joint Statistics of Complex Wavelet Coefficient, Portilla, J. and Simoncelli, E.(2000)](http://www.cns.nyu.edu/pub/lcv/portilla99.pdf)
- [The Heeger-Bergen Pyramid-Based Texture Synthesis Algorithm, Briend, T. et al.(2014)](http://www.ipol.im/pub/art/2014/79/)



