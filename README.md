# MMLineageTracking
Lineage Tracking Algorithm for Mother Machine Using Cells Properties

Please have a look at the [iPython notebook](MMLT_example.ipynb) for more information

The algorithm consists of four main steps.
1. **Loading cell properties**: Each cell is defined by several different properties extracted from the segmentation results. These properties include factors like cell position, length, area, and image channel intensity.

2. **Simulating cell’s futures**: Transitioning from one frame to the next, the physical properties of cells are efficiently simulated to change in geometry under the constraints of the MM. This will generate many possible scenarios, each of which is assigned a prior probability. This probability is determined based on the cell size regulation models such as the adder and sizer models. The parameters necessary to calculate this probability are extracted from the mother cell (which is to track) and the tracked lineages.

3. **Matching with true future**: For each scenario, the array of simulated properties at each frame is matched with the subsequent frame, yielding a soft-max likelihood probability for each simulation. This likelihood is computed from the minimum distance achieved when matching the two high-dimensional arrays, with order constraints imposed since the cells cannot switch positions in the MM. Moreover, this approach also allows the detection of lysis events by permitting skipping in the matching process. The final tracking results are obtained by performing Bayesian inference with the prior and likelihood probability.

4. **Storing tracked lineage in iteration**: Given that the number of simulations scales exponentially with the number of cells that we need to track simultaneously, and the simulation noise increases proportionally to the number of cells (as positional changes accumulate in one direction), we choose to concurrently track only a restricted number of cells and retain the lineage results. This strategy allows us to track more new cells in the subsequent iterations, with updated parameters and remembered lineages. This significantly improves both the efficiency and the precision of the algorithm.


# System Requirements

## Hardware Requirements

The algorithm requires only a standard computer with enough RAM to support the operations defined by a user. For minimal performance, this will be a computer with about 2 GB of RAM. For optimal performance, we recommend a computer with the following specs:

RAM: 16+ GB  
CPU: 4+ cores, 3.3+ GHz/core

The runtimes are highly variable depending on the number of bacteria, frames, and mode of tracking, below are generated using a computer with 16 GB RAM, 8 cores@3.3 GHz.

Time taken to track N cell simultaneously over 61 frames with the adaptive number of division per frame to simulate set to 3:
| Average No. of cells | sizer-adder with skewed model | sizer with skewed model | sizer-adder with unskewed model |
| --- | --- | --- | --- |
|3.5 |	2.51 s $\pm$ 296 ms |	413 ms$\pm$ 4.4 ms |	2.67 s $\pm$ 86.1 ms |
|6.45 |	8.18 s $\pm$ 244 ms |	2.23 s $\pm$ 64.1 ms |	7.49 s $\pm$ 593 ms |
|9.15 |	20.3 s $\pm$ 1.57 s |	7.94 s $\pm$ 102 ms |	21.9 s $\pm$ 642 ms |
|11.92 |	1min 5s $\pm$ 6.01 s |	29.3 s $\pm$ 646 ms |	48.4 s $\pm$ 2.17 s| 
|14.75 |	2min 58s $\pm$ 10.2 s |	1min 35s $\pm$ 466 ms |	2min 26s $\pm$ 12.3 s |
|17.57 |	10min 4s $\pm$ 50.2 s |	6min $\pm$ 16.3 s |	8min 26s $\pm$ 38 s |


## Software Requirements

### OS Requirements

The package development version is tested on *Linux* and *Windows* operating systems. The developmental version of the package has been tested on the following systems:

Linux: Ubuntu 24.04
Windows: Windows 10

### Package dependencies

The versions of software are shown in the setup.yml file
