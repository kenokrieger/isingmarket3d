# 3D Ising Market Model

<img src="https://img.shields.io/github/issues/kenokrieger/isingmarket3d"> <img src="https://img.shields.io/github/commit-activity/m/kenokrieger/isingmarket3d">
<img src="http://qmpy.org/badges/license.svg">

## Outline

This particular model attempts to predict the behaviour of traders in a market
governed by two simple guidelines:

- Do as neighbours do

- Do what the minority does

mathematically speaking is each trader represented by a spin on a three dimensional
grid. The local field of each spin *S*<sub>i</sub> is given by the equation below

<img src="https://github.com/kenokrieger/isingmarket3d/blob/main/.images/local_field.png">

where *J*<sub>ij</sub> = *j* for the nearest neighbours and 0 otherwise. The spins
are updated according to a Heatbath dynamic which reads as follows

<img src="https://github.com/kenokrieger/isingmarket3d/blob/main/.images/spin_updates.png">


The model is thus controlled by the three parameters

- &alpha;, which represents the tendency of the traders to be in the minority

- *j*, which affects how likely it is for a trader to pick up the strategy of its neighbour

- &beta;, which controls the randomness

In each iteration all spins are updated in parallel using the metropolis
algorithm. </br>
(For more details see <a href="https://arxiv.org/pdf/cond-mat/0105224.pdf">
S.Bornholdt, "Expectation bubbles in a spin model of markets: Intermittency from
frustration across scales, 2001"</a>)

## Compiling

To compile the code on Ubuntu you need to have the CUDA development toolkit as well
as a the gcc compiler installed. A simple `make` should be sufficient.
