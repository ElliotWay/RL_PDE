import argparse
import importlib
import os
import sys

import matplotlib.pyplot as plt
import tensorflow as tf
import yaml

import compare
from util import msg, profile, runparams, io

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from rl_pde.emi import DimensionalAdapterEMI, VectorAdapterEMI
from rl_pde.agents import get_agent
from envs import builder as env_builder
from models import builder as model_builder
from util import metadata
from util.param_manager import ArgTreeManager
from util.function_dict import numpy_fn
from util.lookup import get_model_class, get_emi_class, get_model_dims
from util.misc import set_global_seed

valid_solvers = ["advection",
                 "advection_nonuniform",
                 "advection_rk",
                 "advection_fv4",
                 "advection_weno",
                 "compressible",
                 "compressible_rk",
                 "compressible_fv4",
                 "compressible_sdc",
                 "compressible_react",
                 "compressible_sr",
                 "diffusion",
                 "incompressible",
                 "lm_atm",
                 "swe"]


class Pyro(object):
    """
    The main driver to run pyro.
    """

    def __init__(self, solver_name):
        """
        Constructor

        Parameters
        ----------
        solver_name : str
            Name of solver to use
        """

        msg.bold('pyro ...')

        if solver_name not in valid_solvers:
            msg.fail("ERROR: %s is not a valid solver" % solver_name)

        self.pyro_home = os.path.dirname(os.path.realpath(__file__)) + '/'

        # import desired solver under "solver" namespace
        self.solver = importlib.import_module(solver_name)
        self.solver_name = solver_name

        # -------------------------------------------------------------------------
        # runtime parameters
        # -------------------------------------------------------------------------

        # parameter defaults
        self.rp = runparams.RuntimeParameters()
        self.rp.load_params(self.pyro_home + "_defaults")
        self.rp.load_params(self.pyro_home + solver_name + "/_defaults")

        self.tc = profile.TimerCollection()

        self.is_initialized = False

    def initialize_problem(self, problem_name, inputs_file=None, inputs_dict=None,
                           other_commands=None):
        """
        Initialize the specific problem

        Parameters
        ----------
        problem_name : str
            Name of the problem
        inputs_file : str
            Filename containing problem's runtime parameters
        inputs_dict : dict
            Dictionary containing extra runtime parameters
        other_commands : str
            Other command line parameter options
        """

        problem_defaults_file = self.pyro_home + self.solver_name + \
            "/problems/_" + problem_name + ".defaults"

        # problem-specific runtime parameters
        if os.path.isfile(problem_defaults_file):
            self.rp.load_params(problem_defaults_file)

        # now read in the inputs file
        if inputs_file is not None:
            if not os.path.isfile(inputs_file):
                # check if the param file lives in the solver's problems directory
                inputs_file = self.pyro_home + self.solver_name + "/problems/" + inputs_file
                print("using inputs file: ", inputs_file)
                if not os.path.isfile(inputs_file):
                    msg.fail("ERROR: inputs file does not exist")

            self.rp.load_params(inputs_file, no_new=1)

        if inputs_dict is not None:
            for k, v in inputs_dict.items():
                self.rp.params[k] = v

        # and any commandline overrides
        if other_commands is not None:
            self.rp.command_line_params(other_commands)

        # write out the inputs.auto
        self.rp.print_paramfile()

        self.verbose = self.rp.get_param("driver.verbose")
        self.dovis = self.rp.get_param("vis.dovis")

        # -------------------------------------------------------------------------
        # initialization
        # -------------------------------------------------------------------------

        # initialize the Simulation object -- this will hold the grid and
        # data and know about the runtime parameters and which problem we
        # are running
        self.sim = self.solver.Simulation(
            self.solver_name, problem_name, self.rp, timers=self.tc)

        self.sim.initialize()
        self.sim.preevolve()

        plt.ion()

        self.sim.cc_data.t = 0.0

        self.is_initialized = True

    def run_sim(self, agent=None):
        """
        Evolve entire simulation
        """

        if not self.is_initialized:
            msg.fail("ERROR: problem has not been initialized")

        tm_main = self.tc.timer("main")
        tm_main.begin()

        # output the 0th data
        basename = self.rp.get_param("io.basename")
        self.sim.write("{}{:04d}".format(basename, self.sim.n))

        if self.dovis:
            plt.figure(num=1, figsize=(8, 6), dpi=100, facecolor='w')
            self.sim.dovis()

        while not self.sim.finished():
            self.single_step(agent)

        # final output
        if self.verbose > 0:
            msg.warning("outputting...")
        basename = self.rp.get_param("io.basename")
        self.sim.write("{}{:04d}".format(basename, self.sim.n))

        tm_main.end()
        # -------------------------------------------------------------------------
        # final reports
        # -------------------------------------------------------------------------
        if self.verbose > 0:
            self.rp.print_unused_params()
            self.tc.report()

        self.sim.finalize()

        return self.sim

    def single_step(self, agent=None):
        """
        Do a single step
        """

        if not self.is_initialized:
            msg.fail("ERROR: problem has not been initialized")

        # fill boundary conditions
        self.sim.cc_data.fill_BC_all()

        # get the timestep
        self.sim.compute_timestep()

        # evolve for a single timestep
        self.sim.evolve(agent)

        if self.verbose > 0:
            print("%5d %10.5f %10.5f" %
                  (self.sim.n, self.sim.cc_data.t, self.sim.dt))

        # output
        if self.sim.do_output():
            if self.verbose > 0:
                msg.warning("outputting...")
            basename = self.rp.get_param("io.basename")
            self.sim.write("{}{:04d}".format(basename, self.sim.n))

        # visualization
        if self.dovis:
            tm_vis = self.tc.timer("vis")
            tm_vis.begin()

            self.sim.dovis()
            store = self.rp.get_param("vis.store_images")

            if store == 1:
                basename = self.rp.get_param("io.basename")
                plt.savefig("{}{:04d}.png".format(basename, self.sim.n))

            tm_vis.end()

    def __repr__(self):
        """ Return a representation of the Pyro object """
        s = "Solver = {}\n".format(self.solver_name)
        if self.is_initialized:
            s += "Problem = {}\n".format(self.sim.problem_name)
            s += "Simulation time = {}\n".format(self.sim.cc_data.t)
            s += "Simulation step number = {}\n".format(self.sim.n)
        s += "\nRuntime Parameters"
        s += "\n------------------\n"
        s += str(self.rp)
        return s

    def get_var(self, v):
        """
        Alias for cc_data's get_var routine, returns the cell-centered data
        given the variable name v.
        """

        if not self.is_initialized:
            msg.fail("ERROR: problem has not been initialized")

        return self.sim.cc_data.get_var(v)


class PyroBenchmark(Pyro):
    """
    A subclass of Pyro for benchmarking. Inherits everything from pyro, but adds benchmarking routines.
    """

    def __init__(self, solver_name, comp_bench=False,
                 reset_bench_on_fail=False, make_bench=False):
        """
        Constructor

        Parameters
        ----------
        solver_name : str
            Name of solver to use
        comp_bench : bool
            Are we comparing to a benchmark?
        reset_bench_on_fail : bool
            Do we reset the benchmark on fail?
        make_bench : bool
            Are we storing a benchmark?
        """

        super().__init__(solver_name)

        self.comp_bench = comp_bench
        self.reset_bench_on_fail = reset_bench_on_fail
        self.make_bench = make_bench

    def run_sim(self, agent, rtol):
        """
        Evolve entire simulation and compare to benchmark at the end.
        """

        super().run_sim(agent)

        result = 0

        if self.comp_bench:
            result = self.compare_to_benchmark(rtol)

        if self.make_bench or (result != 0 and self.reset_bench_on_fail):
            self.store_as_benchmark()

        if self.comp_bench:
            return result
        else:
            return self.sim

    def compare_to_benchmark(self, rtol):
        """ Are we comparing to a benchmark? """

        basename = self.rp.get_param("io.basename")
        compare_file = "{}/tests/{}{:04d}".format(
            self.solver_name, basename, self.sim.n)
        msg.warning("comparing to: {} ".format(compare_file))
        try:
            sim_bench = io.read(compare_file)
        except IOError:
            msg.warning("ERROR opening compare file")
            return "ERROR opening compare file"

        result = compare.compare(self.sim.cc_data, sim_bench.cc_data, rtol)

        if result == 0:
            msg.success("results match benchmark to within relative tolerance of {}\n".format(rtol))
        else:
            msg.warning("ERROR: " + compare.errors[result] + "\n")

        return result

    def store_as_benchmark(self):
        """ Are we storing a benchmark? """

        if not os.path.isdir(self.solver_name + "/tests/"):
            try:
                os.mkdir(self.solver_name + "/tests/")
            except (FileNotFoundError, PermissionError):
                msg.fail(
                    "ERROR: unable to create the solver's tests/ directory")

        basename = self.rp.get_param("io.basename")
        bench_file = self.pyro_home + self.solver_name + "/tests/" + \
            basename + "%4.4d" % (self.sim.n)
        msg.warning("storing new benchmark: {}\n".format(bench_file))
        self.sim.write(bench_file)


def parse_args():
    """Parse the runtime parameters"""

    p = argparse.ArgumentParser()

    p.add_argument("--make_benchmark",
                   help="create a new benchmark file for regression testing",
                   action="store_true")
    p.add_argument("--compare_benchmark",
                   help="compare the end result to the stored benchmark",
                   action="store_true")

    p.add_argument("solver", metavar="solver-name", type=str, nargs=1,
                   help="name of the solver to use", choices=valid_solvers)
    p.add_argument("problem", metavar="problem-name", type=str, nargs=1,
                   help="name of the problem to run")
    p.add_argument("param", metavar="inputs-file", type=str, nargs=1,
                   help="name of the inputs file")

    p.add_argument("other", metavar="runtime-parameters", type=str, nargs="*",
                   help="additional runtime parameters that override the inputs file "
                   "in the format section.option=value")

    return p.parse_args()


def load_rl_agent():
    arg_manager = ArgTreeManager()
    parser = argparse.ArgumentParser(
        description="Deploy an existing RL agent in an environment. Note that this script also takes various arguments not listed here.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--help-env', default=False, action='store_true',
                        help="Do not test and show the environment parameters not listed here.")
    parser.add_argument('--agent', '-a', type=str,
                        default="log/weno_euler/full/211123_165505/best_1_model_20000.zip",  # order=2 RL agent
                        # default="log/weno_euler/full/211123_165646/best_1_model_19990.zip",  # order=3 RL agent
                        help="Agent to test. Either a file or a string for a standard agent."
                             + " Parameters are loaded from 'meta.[yaml|txt]' in the same directory as the"
                             + " agent file, but can be overriden."
                             + " 'default' uses standard weno coefficients. 'none' forces no agent and"
                             + " only plots the true solution (ONLY IMPLEMENTED FOR EVOLUTION PLOTS).")
    parser.add_argument('--env', type=str, default="weno_euler",
                        help="Name of the environment in which to deploy the agent.")
    parser.add_argument('--model', type=str, default=None,
                        help="Type of model to be loaded. (Overrides the meta file.)")
    parser.add_argument('--emi', type=str, default=None,
                        help="Environment-model interface. (Overrides the meta file.)")
    parser.add_argument('--obs-scale', '--obs_scale', type=str, default='z_score_last',
                        help="Adjustment function to observation. Compute Z score along the last"
                             + " dimension (the stencil) with 'z_score_last', the Z score along every"
                             + " dimension with 'z_score_all', or leave them the same with 'none'.")
    parser.add_argument('--action-scale', '--action_scale', type=str, default=None,
                        help="Adjustment function to action. Default depends on environment."
                             + " 'softmax' computes softmax, 'rescale_from_tanh' scales to [0,1] then"
                             + " divides by the sum of the weights, 'none' does nothing.")
    parser.add_argument('--log-dir', type=str, default=None,
                        help="Directory to place log file and other results. Default is test/env/agent/timestamp.")
    parser.add_argument('--seed', type=int, default=1,
                        help="Set random seed for reproducibility.")
    parser.add_argument('--no-state', '--no_state', default=True, dest='plot_state',
                        action='store_false',
                        help="Override the default and do not plot the state of the environment.")
    parser.add_argument('--plot-actions', '--plot_actions', default=False, action='store_true',
                        help="Plot the agent's actions.")
    parser.add_argument('--animate', default=False, action='store_true',
                        help="Enable animation mode. Plot the state at every timestep,"
                             + " and keep the axes fixed across every plot.")
    parser.add_argument('--plot-error', '--plot_error', default=False, action='store_true',
                        help="Plot the error between the agent and the solution. Combines with evolution-plot.")
    parser.add_argument('--plot-tv', '--plot_tv', default=False, action='store_true',
                        help="Plot the total variation vs time for an episode.")
    parser.add_argument('--evolution-plot', '--evolution_plot', default=False, action='store_true',
                        help="Instead of usual rendering create 'evolution plot' which plots several states on the"
                             + " same plot in increasingly dark color.")
    parser.add_argument('--convergence-plot', '--convergence_plot', nargs='*', type=int,
                        default=None,
                        help="Do several runs with different grid sizes to create a convergence plot."
                             + " Overrides the --num-cells parameter and sets the --analytical flag."
                             + " Use e.g. '--convergence-plot' to use the default grid sizes."
                             + " Use e.g. '--convergence-plot A B C D' to specify your own.")
    parser.add_argument('--output-mode', '--output_mode', default=['plot'], nargs='+',
                        help="Type of output from the test. Default 'plot' creates the usual plot"
                             + " files. 'csv' puts the data that would be used for a plot in a csv"
                             + " file. Currently 'csv' is not implemented for evolution plots."
                             + " Multiple modes can be used at once, e.g. --output-mode plot csv.")
    parser.add_argument('--repeat', type=str, default=None,
                        help="Load all of the parameters from a previous test's meta file to run a"
                             + " similar or identical test. Explicitly passed parameters override"
                             + " loaded paramters.")
    parser.add_argument('-y', '--y', default=False, action='store_true',
                        help="Choose yes for any questions, namely overwriting existing files. Useful for scripts.")
    parser.add_argument('-n', '--n', default=False, action='store_true',
                        help="Choose no for any questions, namely overwriting existing files. Useful for scripts. Overrides the -y option.")

    arg_manager.set_parser(parser)
    env_arg_manager = arg_manager.create_child("e", long_name="Environment Parameters")
    env_arg_manager.set_parser(env_builder.get_env_arg_parser())
    # Testing has 'model parameters', but these are really intended to be loaded from a file.
    model_arg_manager = arg_manager.create_child("m", long_name="Model Parameters")
    model_arg_manager.set_parser(lambda args: model_builder.get_model_arg_parser(args.model))

    args, rest = arg_manager.parse_known_args()

    if args.repeat is not None:
        _, extension = os.path.splitext(args.repeat)

        if extension == '.yaml':
            open_file = open(args.repeat, 'r')
            args_dict = yaml.safe_load(open_file)
            open_file.close()
            # y and n are special, and should never be loaded.
            args_dict['y'] = args.y
            args_dict['n'] = args.n
            arg_manager.load_from_dict(args_dict)
            args = arg_manager.args
        else:
            # Original meta.txt format.
            metadata.load_to_namespace(args.repeat, arg_manager)

    for mode in args.output_mode:
        if mode not in ['plot', 'csv']:
            raise Exception(f"{mode} output mode not recognized.")

    # Convergence plots have different defaults.
    if args.convergence_plot is not None:
        if not arg_manager.check_explicit('e.init_type'):
            args.e.init_type = 'gaussian'
            args.e.time_max = 0.05
            args.e.C = 0.5
            print("Convergence Plot default environment loaded.")
            print("(Gaussian, t_max = 0.05, C = 0.5)")
            # Mark these arguments as explicitly specified so cannot be overridden
            # by a loaded agent.
            # (Do we need to do this? We do for the old system that loads all the parameters, but
            # the new system only loads the model parameters and a few others, not the environment
            # parameters. Does it make more sense to leave them as implicit defaults so they can be
            # loaded by an intentionally loaded environment file?)
            arg_manager.set_explicit('e.init_type', 'e.time_max', 'e.C')
            # The old way to mark explicit. Remove if we don't need backwards compatability with
            # old meta files.
            sys.argv += ['--init-type', 'gaussian', '--time-max', '0.05', '--C', '0.5']
        if not arg_manager.check_explicit('e.rk'):
            args.e.rk = 'rk4'
            arg_manager.set_explicit('e.rk')
            sys.argv += ['--rk', '4']
        if not arg_manager.check_explicit('e.fixed_timesteps'):
            args.e.fixed_timesteps = False
            arg_manager.set_explicit('e.fixed_timesteps')
            sys.argv += ['--variable-timesteps']

    env_builder.set_contingent_env_defaults(args, args.e, test=True)
    model_builder.set_contingent_model_defaults(args, args.m, test=True)

    env_action_type = env_builder.env_action_type(args.env)
    dims = env_builder.env_dimensions(args.env)

    # Load basic agent, if one was specified.
    agent = get_agent(args.agent, order=args.e.order, action_type=env_action_type, dimensions=dims)
    # If the returned agent is None, assume it is the file name of a real agent.
    # If a real agent is specified, load the model parameters from its meta file.
    # This only overrides arguments that were not explicitly specified.
    if agent is None:
        if not os.path.isfile(args.agent):
            raise Exception("Agent file \"{}\" not found.".format(args.agent))

        model_file = os.path.abspath(args.agent)
        model_directory = os.path.dirname(model_file)
        meta_file = os.path.join(model_directory, metadata.META_FILE_NAME)
        if os.path.isfile(meta_file):
            open_file = open(meta_file, 'r')
            args_dict = yaml.safe_load(open_file)
            open_file.close()

            # action scale and obs_scale ought to be model parameters.
            arg_manager.load_keys(args_dict, ['model', 'emi', 'action_scale', 'obs_scale'])
            # Should order be a base level parameter? It affects both model and environment.
            env_arg_manager.load_keys(args_dict['e'], ['order'])

            model_arg_manager.load_from_dict(args_dict['m'])
        else:
            meta_file = os.path.join(model_directory, metadata.OLD_META_FILE_NAME)
            if not os.path.isfile(meta_file):
                raise Exception("Meta file \"{}\" for agent not found.".format(meta_file))

            metadata.load_to_namespace(meta_file, arg_manager,
                                       ignore_list=['log_dir', 'ep_length', 'time_max', 'timestep',
                                                    'num_cells', 'min_value', 'max_value', 'C', 'fixed_timesteps',
                                                    'reward_mode'])

    set_global_seed(args.seed)

    if args.convergence_plot is None:
        env = env_builder.build_env(args.env, args.e, test=True)
    else:
        env_manager_copy = env_arg_manager.copy()
        env_args = env_manager_copy.args
        env_args.analytical = True  # Compare to analytical solution (preferred)
        # env_args.analytical = False # Compare to WENO (necessary when WENO isn't accurate either)
        if env_args.reward_mode is not None and 'one-step' in env_args.reward_mode:
            print("Reward mode switched to 'full' instead of 'one-step' for convergence plots.")
            env_args.reward_mode = env_args.reward_mode.replace('one-step', 'full')
        if len(args.convergence_plot) > 0:
            convergence_grid_range = args.convergence_plot
        elif dims == 1:
            # The original grid sizes included 81 instead of 82, but the parity of the grid size
            # has peculiar affects on certain initial conditions (namely smooth_sine), so we switch
            # the 81 to 82 so all grid sizes are even.
            convergence_grid_range = [64, 82, 108, 128, 144, 192, 256]
            # convergence_grid_range = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
            # convergence_grid_range = (2**np.linspace(6.0, 8.0, 50)).astype(np.int)
        elif dims == 2:
            convergence_grid_range = [32, 64, 128, 256]

        # Set ep_length and timestep based on number of cells and time_max.
        env_args.ep_length = None
        env_args.timestep = None
        if env_args.C is None:
            env_args.C = 0.1

        conv_envs = []
        conv_env_args = []
        for nx in convergence_grid_range:
            specific_env_copy = env_manager_copy.copy()
            env_args = specific_env_copy.args
            env_args.num_cells = nx

            env_builder.set_contingent_env_defaults(args, env_args, test=True,
                                                    print_prefix=f"{nx}: ")

            conv_envs.append(env_builder.build_env(args.env, env_args, test=True))
            conv_env_args.append(env_args)
        env = conv_envs[0]

    if agent is None:
        obs_adjust = numpy_fn(args.obs_scale)
        action_adjust = numpy_fn(args.action_scale)

        model_cls = get_model_class(args.model)
        emi_cls = get_emi_class(args.emi)
        model_dims = get_model_dims(args.model)

        if model_dims < dims:
            if model_dims == 1:
                emi = DimensionalAdapterEMI(emi_cls, env, model_cls, args,
                                            obs_adjust=obs_adjust, action_adjust=action_adjust)
            else:
                raise Exception("Cannot adapt {}-dimensional model to {}-dimensional environment."
                                .format(model_dims, dims))
        elif args.env == 'weno_euler':
            emi = VectorAdapterEMI(emi_cls, env, model_cls, args, obs_adjust=obs_adjust, action_adjust=action_adjust)
        else:
            emi = emi_cls(env, model_cls, args, obs_adjust=obs_adjust, action_adjust=action_adjust)

        emi.load_model(args.agent)
        agent = emi.get_policy()

    return agent


if __name__ == "__main__":
    args_pyro = parse_args()
    agent = load_rl_agent()

    if args_pyro.compare_benchmark or args_pyro.make_benchmark:
        pyro = PyroBenchmark(args_pyro.solver[0],
                             comp_bench=args_pyro.compare_benchmark,
                             make_bench=args_pyro.make_benchmark)
    else:
        pyro = Pyro(args_pyro.solver[0])

    pyro.initialize_problem(problem_name=args_pyro.problem[0],
                            inputs_file=args_pyro.param[0],
                            other_commands=args_pyro.other)
    pyro.run_sim(agent)
