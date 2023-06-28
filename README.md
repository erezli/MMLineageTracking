# MMLineageTracking
Lineage Tracking Algorithm for Mother Machine Using Cells Properties

Please have a look at the [iPython notebook](MM_Cell_Lineage_Tracking_Example.ipynb) for more information

The algorithm consists of four main steps.
1. **Loading cell properties**: Each cell is defined by several different properties extracted from the segmentation results. These properties include factors like cell position, length, area, and image channel intensity.

2. **Simulating cell’s futures**: Transitioning from one frame to the next, the physical properties of cells are efficiently simulated to change in geometry under the constraints of the MM. This will generate many possible scenarios, each of which is assigned a prior probability. This probability is determined based on the cell size regulation models such as the adder and sizer models. The parameters necessary to calculate this probability are extracted from the mother cell (which is to track) and the tracked lineages.

3. **Matching with true future**: For each scenario, the array of simulated properties at each frame is matched with the subsequent frame, yielding a soft-max likelihood probability for each simulation. This likelihood is computed from the minimum distance achieved when matching the two high-dimensional arrays, with order constraints imposed since the cells cannot switch positions in the MM. Moreover, this approach also allows the detection of lysis events by permitting skipping in the matching process. The final tracking results are obtained by performing Bayesian inference with the prior and likelihood probability.

4. **Storing tracked lineage in iteration**: Given that the number of simulations scales exponentially with the number of cells that we need to track simultaneously, and the simulation noise increases proportionally to the number of cells (as positional changes accumulate in one direction), we choose to concurrently track only a restricted number of cells and retain the lineage results. This strategy allows us to track more new cells in the subsequent iterations, with updated parameters and remembered lineages. This significantly improves both the efficiency and the precision of the algorithm.