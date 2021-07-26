# nav-sims
Agent-based simulations of olfactory navigation


## Installation

All code runs with Python3 and standard Python libraries.

## Usage

We can use the code in this repository to simulate olfactory navigation strategies as well as turbulent odor plumes themselves. 
Users have the option of using recorded video plumes as well as simulating plumes using a computationally efficient model of drifting
and diffusing filaments of odor due to (Farrell et al. 2002). 

Simulations are run with run scripts like those provided, that call wrappers which in turn call model scripts and environment scripts. The navigational model
provided uses odor intermittency (the proportion of time the signal can be detected) and odor hit frequency to bias motion upwind in an attempt to get to the source
via a biased random walk. 

It is recommended these simulations are run on a cluster, where perhaps a Job ID number can be provided as a system argument to the run script in order to seed
random numbers. 
