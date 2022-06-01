# RL_PDE

Source code for **Backpropagation through Time and Space: Learning Numerical Methods with Multi-Agent Reinforcement Learning** (https://arxiv.org/abs/2203.08937).
Authors: Elliot Way, Dheeraj S.K. Kapilavai, Yiwei Fu, Lei Yu

The development of this software was partially supported by the Defense Advanced Research Projects Agency (DARPA) under Agreement No. HR00112090063. Approved for public release; distribution is unlimited.

It is worth noting that this is code developed during research; expect only a minimum of polish and code that goes nowhere because that experiment didn't work.

## Installation

Python 3.7 is required. conda or venv can both connect to a specific version of Python.

Required packages can be installed from the requirements file with
```
pip install -r requirements.txt
```

### Notable Requirements

This code is written with Tensorflow 1.15, which causes challenges. This is the reason why Python 3.7 is required.
Tensorflow 1.15 also apparently requires protobuf<=3.20, which needed to be specified explicitly in the requirements file.

Early development in this project relied on [Stable Baselines 2](https://github.com/hill-a/stable-baselines).
Many similarities to Stable Baselines have carried over into the current version of this repository,
but installing Stable Baselines itself is not required to run the main experiments.
However, old experiments are still present. To run those, installing Stable Baselines is required with e.g.
```
pip install stable_baselines
```
See http://stable-baselines.readthedocs.io/.

## Running Experiments

Train an agent with
```
python run_train.py
```

then test it with
```
python run_test.py --agent last/model_best.zip
```

`run_train.py` and `run_test.py` take a wealth of optional parameters. More information about these can be inspected with e.g.
```
python run_train.py -h
```
However, all parameters have defaults based on the experiments in the paper.
Simply `python run_train.py` will train an agent with the 1D Burgers equation with identical parameters to those used in the paper.

Additional parameter information can be accessed with `--help-model` for model parameters and `--help-env` for environment parameters.
Model parameters are restricted by the type of model, so
```
python run_train.py --model full --help-model
```
will show only the parameters relevant to the "full" type of model.
"full" corresponds to the BPTTS model described in the paper; the results of the paper do not use any other type of model.

### Experiment Outputs

`run_train.py` and `run_test.py` collect files into a log directory specified by `--log-dir` or using a unique default based on the current timestamp.
For convenience, a symbolic link `last` links to the log directory of the most recent experiment.

Runs with both `run_train.py` and `run_test.py` will have:

`meta.yaml` contains all of the parameters used to run the experiment, as well as other meta information in the comment header,
such as whether the experiment ended unexpectedly.

`progress.csv` contains data collected from the experiment in comma-separated format. In practice, `run_test.py` only uses `progress.csv` for some data.

`log.txt` was intended to contain printed output messages; in practice it only contains the data from progress.csv in a more human-readable format.

For training runs with `run_train.py`, notable other files are `model_best.zip`, the saved model with the best testing performance;
`model_final.zip`, the model saved after the final training iteration; and the plots in `summary_plots` which plot metrics against training iterations.


## Source Code Summary

[run_train.py](run_train.py) and [run_test.py](run_test.py) are the main entry points.
[param_sweep_template.py](param_sweep_template.py) runs a basic parameter sweep over other commands to create many experiments at once.
It is intended for use with `run_train.py` and `run_test.py` but can be used with any command.

[scripts/](scripts) contains a variety of scripts to use with the output of experiments, mainly consolidating information into plots.  
Use e.g. `python scripts/action_comparison_plot.py -h` for a description of what each script does.

Files in [envs/](envs) encode the structured environments used to represent modelling a PDE.
The original design was based on the OpenAI Gym interface, though this drifted over time and the current environments do not match that interface.

Files in [models/](models) encode the models used for training an agent.

Files in [rl_pde/](rl_pde) are for generally running experiments.
In particular, [rl_pde/emi/](rl_pde/emi) is concerned with the "Environment-Model Interface" between the environment and the model used for training,
and [rl_pde/agents/](rl_pde/agents) is concerned with the agents that add control into that interface.

Files is [util/](util) are miscellaneous functions that don't belong anywhere else.


