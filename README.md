The "curse of dimensionality" is one of the main engineering challenges in machine learning due to the exponential cost of exact algorithms. This is typically addressed by generalization and approximation of state or action space. Deep Q-Network (DQN) proposed by Mnih el al. (2015), attempts to overcome the dependency in the size of the state space by learning an approximate value function. However, this scheme does not solve the dependency in the action space size. 
For instance, agents operating in text domains typically handle very large discrete action space in addition to a large state space. This work suggests leveraging weak prior knowledge to allow the agent to learn to consider only a small subset of actions at each state. Text-based games (TBG) are a natural candidate for testing this hypothesis, thus a new framework was constructed to allow DQN agents to interact with the popular title "Zork I - the great underground Empire".
Experiments show that a speculative restriction scheme scales well and converges significantly faster when the action space is very large while producing stable policies compared to vanilla DQN agents which attempt to learn over the entire action space.
Note: The code in this repository is a somewhat crude revision of the original DQN agent, adapted to the domain of TBG.

Installation instructions
-------------------------
The installation requires Linux with apt-get.

Note: In order to run the GPU version of DQN, you should additionally have the
NVIDIA® CUDA® (version 5.5 or later) toolkit installed prior to the Torch
installation below.
This can be downloaded from https://developer.nvidia.com/cuda-toolkit
and installation instructions can be found in
http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux

To install in a subdirectory called 'torch', it should be enough to run

    ./install_dependencies.sh

from the base directory of the package.

Note: The above install script will install the following packages via apt-get:
build-essential, gcc, g++, cmake, curl, libreadline-dev, git-core, libjpeg-dev,
libpng-dev, ncurses-dev, imagemagick, unzip

Training DQN on Zork
---------------------------
    ./run_gpu zork <scenario> <agent_type> [GPU_ID]

Note: On a system with more than one GPU, DQN training can be launched on a
specified GPU by setting the environment variable GPU_ID. If GPU_ID is not specified, the first available GPU (ID 0) will be used by default.

Options
-------
Options to DQN are set within run_gpu, check file for additional settings.
