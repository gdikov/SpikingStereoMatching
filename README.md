# Spiking Stereo Matching

**Update:**
The SNN architecture presented here is reimplemented in a cleaner and more efficient way in 
[hybrid-stereo-matching](https://github.com/gdikov/hybrid-stereo-matching). Although the latter repository is dedicated
to a slightly different task, the SNN is essentially the same and all experiments from [1] can be reproduced 
using suitable configuration files. For more details on how to do this, see the subsection *SNN only* in
*Running custom experiments* described in the README there.

### Overview

This repository contains the original code and data needed to reproduce the experiments from my Bachelor thesis on 
spiking neural network (SNN) for real-time event-based stereo matching using dynamic vision sensors [4] and the SpiNNaker [3]
neuromorphic hardware. The design is inspired from the guidelines for a cooperative network presented in [2] and 
is thoroughly described in [1]. The implementation uses the sPyNNaker tool chain which provides a PyNN-like API.
The network has been tested both on a local SpiNNaker machine and on the HBP platform.  
 
### Experiments and results

In order to reproduce an experiment just select the desired in `main.py`. If you are interested in running the network on
your own dataset, then see in `examples` one of the existing experiments and add your own `custom_experiment.py` 
accordingly. The input and output data is normally stored under `data/input` and `spikes` respectively. 

In addition to the evaluation presented in [1], you can find animations with the network output at 
[https://figshare.com/s/0d9fb146149b832ed8ec](https://figshare.com/s/0d9fb146149b832ed8ec) 
(thanks to [Christoph Richter](https://github.com/tophensen)). 

### Acknowledgements

Many thanks to [Christoph Richter](https://github.com/tophensen) for the multiple fruitful discussions and invaluable 
support throughout my thesis. 
 
### References

[1] Dikov, G., Firouzi, M., Roehrbein, F., Conradt, J., Richter, C.: Spiking cooperative stereo-matching at 2 ms 
latency with neuromorphic hardware. Proc. Biomimetic and Biohybrid Systems, 119-137 (2017)

[2] Marr, D., Poggio, T.: Cooperative computation of stereo disparity. Science 194(4262), 283–287 (1976)

[3] Furber, S.B., Galluppi, F., Temple, S., Plana, L.A.: The SpiNNaker project. Proc. IEEE 102(5), 652–665 (2014)

[4] Lichtsteiner, P., Posch, C., Delbruck, T.: A 128 × 128 120 db 15 μs latency asynchronous temporal contrast 
vision sensor. IEEE J. Solid State Circ. 43(2), 566–576 (2008)