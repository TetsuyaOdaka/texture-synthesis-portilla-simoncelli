# texture-sysnthesis-polltila-simoncelli

Python3 implementation of Polltila and Simoncelli(2000)'s texture synthesis.  
  
[Parametric Texture Model Based on Joint Statistics of Complex Wavelet Coefficient, Portilla, J. and Simoncelli, E.(2000) ](http://www.cns.nyu.edu/pub/lcv/portilla99.pdf)  

Basically this is a port of [Matlab programs of NYU](https://github.com/LabForComputationalVision/textureSynth).

Especially "steerable pyramid" is based on Briand et al.(2014).  

[The Heeger-Bergen Pyramid-Based Texture Synthesis Algorithm, Briand,T. et al. (2014)](http://www.ipol.im/pub/art/2014/79/)

I have already published the [python3 implementation of steerable pyramid](https://github.com/TetsuyaOdaka/SteerablePyramid/) by their equations.


## Results
Attentions !! All sample images are from [NYU websites](http://www.cns.nyu.edu/~lcv/texture/).   


### Gray Scale
### Original image
<img src="https://github.com/TetsuyaOdaka/texture-synthesis-portilla-simoncelli/blob/master/samples/bark.jpg" alt="texture synthesis">  

### Synthesized image
<img src="https://github.com/TetsuyaOdaka/texture-synthesis-portilla-simoncelli/blob/master/samples/bark-out.png" alt="texture synthesisd">  


### Color Version
### Original image
<img src="https://github.com/TetsuyaOdaka/texture-synthesis-portilla-simoncelli/blob/master/samples/radish.jpg" alt="texture synthesis">  

### Synthesized image
<img src="https://github.com/TetsuyaOdaka/texture-synthesis-portilla-simoncelli/blob/master/samples/radish-out.png" alt="texture synthesis">  
