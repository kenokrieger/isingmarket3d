# 3D Ising Market Model

<img src="https://img.shields.io/github/issues/kenokrieger/isingmarket3d"> <img src="https://img.shields.io/github/commit-activity/m/kenokrieger/isingmarket3d">
<img src="http://qmpy.org/badges/license.svg">

## Outline

This particular model attempts to predict the behavior of traders in a market
governed by two simple guidelines:

- Do as neighbors do

- Do what the minority does

mathematically speaking is each trader represented by a spin on a three dimensional
grid. The local field of each spin *S*<sub>i</sub> is given by the equation below

<img src="https://github.com/kenokrieger/isingmarket3d/blob/main/images/local_field.png" alt="field equation" height="100">

where *J*<sub>ij</sub> = *j* for the nearest neighbors and 0 otherwise. The spins
are updated according to a Heatbath dynamic which reads as follows

<img src="https://github.com/kenokrieger/isingmarket3d/blob/main/images/spin_updates.png" alt="Heatbath equation" height="100">


The model is thus controlled by the three parameters

- &alpha;, which represents the tendency of the traders to be in the minority

- *j*, which affects how likely it is for a trader to pick up the strategy of its neighbor

- &beta;, which controls the randomness

(For more details see <a href="https://arxiv.org/pdf/cond-mat/0105224.pdf">
S.Bornholdt, "Expectation bubbles in a spin model of markets: Intermittency from
frustration across scales, 2001"</a>)

## Implementation

### 3D Metropolis Algorithm

The main idea behind the metropolis algorithm is to split the main lattice into
two sub-lattices with half of the original grid width. You can think of these lattices
as tiles on a chessboard (see figure below).</br>
<img src="https://github.com/kenokrieger/isingmarket3d/blob/main/images/metropolis3d.png" alt="3d metropolis algorithm" height="350"> </br>
Each black or white tile at position p = (row, col, lattice_id) can be assigned
an individual index in a 1 dimensional array.
```c++
long long index = lattice_id * grid_width / 2 * grid_height + row * grid_width / 2 + col;
```
All six neighbors of the tile can be found in the array of the opposite color. This
has the advantage, that the spins in one array can be updated while the ones
in the opposite array remain constant.
```c++
template <bool is_black>
void update_spins(type tiles, const type opposite_tiles);
```
The indices of the nearest neighbors are as follows
```c++
int lower_neighbor_row = (row + 1 < grid_height) ? row + 1 : 0;
int upper_neighbor_row = (row - 1 >= 0) ? row - 1: grid_height - 1;
int right_neighbor_col = (col + 1 < grid_width) ? col + 1 : 0;
int left_neighbor_col = (col - 1 >= 0) ? col - 1: grid_width - 1;
int front_neighbor_lattice = (lattice_id - 1 >= 0) ? lattice_id - 1: grid_depth - 1;
int back_neighbor_lattice = (lattice_id + 1 <= grid_depth - 1) ? lattice_id + 1: 0;
```
where neighbors in different rows will always be in the same column as the updated
spin and neighbors in different columns will be in the same row. Only of of the
the ```right_neighbor_col``` or ```left_neighbor_col```is going to be used depending
on row and lattice id parity. One neighbor is always going to have the same index
as the updated spin. Which results in 6 total neighbors.

## Compiling

To compile the code on Ubuntu you need to have the CUDA development toolkit as well
as a the gcc compiler installed. If your system is configured correctly `make`
should produce the executable "ising3d".

## Sources

The main foundation for this project is the 2d implementation of the ising model
by <a href="https://github.com/romerojosh">Joshua Romero</a>.
