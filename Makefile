# Makefile for the figures needed for the paper.
# This file requires that the RL_AGENT defined below already exists.
# This file does not compile source code in anyway.

ALL_INITS=$(ANALYTICAL_INITS) $(OTHER_INITS)
ANALYTICAL_INITS=smooth_sine accelshock gaussian smooth_rare
ANALYTICAL_EXCEPT_SMOOTH_SINE=accelshock gaussian smooth_rare
OTHER_INITS=tophat rarefaction sine line shock para sawtooth random

#ORDER=2
#TEST_DIR=fig_test_order2
#FIG_DIR=figures_order2
#RL_AGENT="agents/order2/agent.zip"
#TRAIN_DIR="agents/order2"

#ORDER=3
#TEST_DIR=fig_test_order3
#FIG_DIR=figures_order3
#RL_AGENT="agents/smooth_sine_order3/agent.zip"
#TRAIN_DIR="agents/smooth_sine_order3/"
#TRAIN_DIR="log/weno_burgers/full/order3_arch_sweep_again/layers_32_32/"

ORDER=3
TEST_DIR=fig_test_raresched
FIG_DIR=figures_raresched
RL_AGENT=log/weno_burgers/full/rare_schedule/model_best.zip
TRAIN_DIR=log/weno_burgers/full/rare_schedule/


RUN_TEST=python run_test.py -y --animate --output-mode csv plot $\
		--plot-tv --plot-l2 --plot-error --plot-actions --evolution-plot $\
		--rk rk4 --fixed-timesteps $\
		--order $(ORDER)

all: plots
plots: state conv tv l2 action animation training

state: $(ALL_INITS:%=%_state)

STATE_SHORTCUTS=$(ALL_INITS:%=%_state)
$(STATE_SHORTCUTS): %_state: $(FIG_DIR)/states/%.png

END_STATE=burgers_state_step500.csv
STATE_SCRIPT=scripts/combine_state_plots.py

ALL_AGENTS=analytical weno rl
A_AGENTS=analytical rl
O_AGENTS=weno rl

$(FIG_DIR)/states/smooth_sine.png: $(STATE_SCRIPT) $(ALL_AGENTS:%=$(TEST_DIR)/smooth_sine/%/$(END_STATE))
	python $^ --labels "true solution" "WENO solution" "RL solution" \
		--title "RL solution at t=0.2" --output $@

define A_STATE_RULE
$$(FIG_DIR)/states/$(init).png: $$(STATE_SCRIPT) $$(A_AGENTS:%=$$(TEST_DIR)/$(init)/%/$$(END_STATE))
	python $$^ --labels "true solution" "RL solution" \
		--title "RL solution at t=0.2" --output $$(FIG_DIR)/states/$(init).png
endef
$(foreach init,$(ANALYTICAL_EXCEPT_SMOOTH_SINE), $(eval $(A_STATE_RULE)))

define O_STATE_RULE
$$(FIG_DIR)/states/$(init).png: $$(STATE_SCRIPT) $$(O_AGENTS:%=$$(TEST_DIR)/$(init)/%/$$(END_STATE))
	python $$^ --labels "WENO solution" "RL solution" \
		--title "RL solution at t=0.2" --output $(FIG_DIR)/states/$(init).png
endef
$(foreach init,$(OTHER_INITS), $(eval $(O_STATE_RULE)))


states=$(ANALYTICAL_INITS:%=$(TEST_DIR)/%/rl/$(END_STATE))
$(states): $(TEST_DIR)/%/rl/$(END_STATE):
	$(RUN_TEST) --analytical --init-type $* --agent $(RL_AGENT) --log-dir $(TEST_DIR)/$*/rl
states=$(ANALYTICAL_INITS:%=$(TEST_DIR)/%/weno/$(END_STATE))
$(states): $(TEST_DIR)/%/weno/$(END_STATE):
	$(RUN_TEST) --analytical --init-type $* --agent weno --log-dir $(TEST_DIR)/$*/weno
states=$(ANALYTICAL_INITS:%=$(TEST_DIR)/%/analytical/$(END_STATE))
$(states): $(TEST_DIR)/%/analytical/$(END_STATE):
	$(RUN_TEST) --analytical --init-type $* --agent weno --log-dir $(TEST_DIR)/$*/analytical --follow-solution

states=$(OTHER_INITS:%=$(TEST_DIR)/%/rl/$(END_STATE))
$(states): $(TEST_DIR)/%/rl/$(END_STATE):
	$(RUN_TEST) --init-type $* --agent $(RL_AGENT) --log-dir $(TEST_DIR)/$*/rl
states=$(OTHER_INITS:%=$(TEST_DIR)/%/weno/$(END_STATE))
$(states): $(TEST_DIR)/%/weno/$(END_STATE):
	$(RUN_TEST) --init-type $* --agent weno --log-dir $(TEST_DIR)/$*/weno

# All other test files are considered to depend on the end state files.
# If end state files change then all other files for that test run have also changed.
# It's simpler to do it this way than to add multiple targets for the above rules.
STATE_DIRS=$(foreach init,$(ANALYTICAL_INITS),$(foreach agent,$(ALL_AGENTS),$(TEST_DIR)/$(init)/$(agent)))$\
	   $(foreach init,$(OTHER_INITS),$(foreach agent,$(O_AGENTS),$(TEST_DIR)/$(init)/$(agent)))
progress_files=$(STATE_DIRS:%=%/progress.csv)
$(progress_files): %/progress.csv: %/$(END_STATE)
last_actions=$(STATE_DIRS:%=%/action_step499.csv)
$(last_actions): %/action_step499.csv: %/$(END_STATE)

tv: $(ALL_INITS:%=%_tv)

TV_SHORTCUTS=$(ALL_INITS:%=%_tv)
$(TV_SHORTCUTS): %_tv: $(FIG_DIR)/tv/%_tv.png

TV_SCRIPT=scripts/combine_time_plots.py
TV_AGENTS=rl weno

$(FIG_DIR)/tv/%_tv.png: $(TV_SCRIPT) $(foreach agent,$(TV_AGENTS),$(TEST_DIR)/%/$(agent)/progress.csv)
	python $^ --labels "RL" "WENO" --ycol tv --title "Total Variation" --output $@

l2: $(ANALYTICAL_INITS:%=%_l2)

L2_SHORTCUTS=$(ANALYTICAL_INITS:%=%_l2)
$(L2_SHORTCUTS): %_l2: $(FIG_DIR)/l2/%_l2.png

L2_SCRIPT=scripts/combine_time_plots.py
L2_AGENTS=rl weno

$(FIG_DIR)/l2/%_l2.png: $(L2_SCRIPT) $(foreach agent,$(L2_AGENTS),$(TEST_DIR)/%/$(agent)/progress.csv)
	python $^ --labels "RL" "WENO" --ycol l2 --yscale log --ylabel L2 --title "L2 Error" --output $@


action: action_plot action_comparison

ACTION_PLOT_SHORTCUTS=$(ALL_INITS:%=%_action_plot)
action_plot: $(ACTION_PLOT_SHORTCUTS)
$(ACTION_PLOT_SHORTCUTS): %_action_plot: $(FIG_DIR)/actions/%_actions.png

ACTION_PLOT_SCRIPT=scripts/combine_action_plots.py
ACTION_AGENTS=rl weno

$(FIG_DIR)/actions/%_actions.png: $(ACTION_PLOT_SCRIPT) \
		$(foreach agent,$(ACTION_AGENTS),$(TEST_DIR)/%/$(agent)/action_step499.csv)
	python $^ --labels "RL" "WENO" --title "Actions at t=1.9996" --output $@

ACTION_COMP_SHORTCUTS=$(ALL_INITS:%=%_action_comparison)
action_comparison: $(ACTION_COMP_SHORTCUTS)
$(ACTION_COMP_SHORTCUTS): %_action_comparison: $(FIG_DIR)/actions/%_comparison.png

ACTION_COMP_SCRIPT=scripts/action_comparison_plot.py

# Set dependency as penultimate action instead of as all actions.
$(FIG_DIR)/actions/%_comparison.png: $(ACTION_COMP_SCRIPT) \
		$(foreach agent,$(ACTION_AGENTS),$(TEST_DIR)/%/$(agent)/action_step499.csv)
	python $(ACTION_COMP_SCRIPT) --first-actions $(TEST_DIR)/$*/rl/action_step*.csv \
		--second-actions $(TEST_DIR)/$*/weno/action_step*.csv --xname RL --yname WENO \
		--title "Action Comparison" --output $@


ANIMATION_SHORTCUTS=$(ALL_INITS:%=%_animation)
animation: $(FIG_DIR)/animation $(ANIMATION_SHORTCUTS)
$(FIG_DIR)/animation:
	mkdir $(FIG_DIR)/animation

$(ANIMATION_SHORTCUTS): %_animation: $(FIG_DIR)/animation/%.gif
# Set depedency as last state instead of as all states.
$(FIG_DIR)/animation/%.gif: $(TEST_DIR)/%/rl/$(END_STATE)
	convert -loop 0 $(TEST_DIR)/$*/rl/burgers_state_step*.png \
		-set delay '%[fx:t==(n-1) || t==0 ? 100 : 10]' $@


convergence: conv
conv: smooth_sine_conv gaussian_conv

CONV_SCRIPT=scripts/combine_convergence_plots.py
CONV_AGENTS=rl weno

RUN_CONV=python run_test.py --convergence-plot -y --order $(ORDER) --rk rk4 --variable-timesteps

smooth_sine_conv: $(FIG_DIR)/convergence/smooth_sine_0_05.png $(FIG_DIR)/convergence/smooth_sine_0_1.png
$(FIG_DIR)/convergence/smooth_sine_0_05.png: \
       		$(CONV_SCRIPT) $(CONV_AGENTS:%=$(TEST_DIR)/convergence/smooth_sine_0_05/%/progress.csv)
	python $^ --poly 3 5 --labels "RL" "WENO" --title "Convergence on sine" \
		--output $(FIG_DIR)/convergence/smooth_sine_0_05.png
$(FIG_DIR)/convergence/smooth_sine_0_1.png: \
		$(CONV_SCRIPT) $(CONV_AGENTS:%=$(TEST_DIR)/convergence/smooth_sine_0_1/%/progress.csv)
	python $^ --labels "RL" "WENO" --title "Convergence on sine" \
		--output $(FIG_DIR)/convergence/smooth_sine_0_1.png

gaussian_conv: $(FIG_DIR)/convergence/gaussian_0_05.png $(FIG_DIR)/convergence/gaussian_0_1.png
$(FIG_DIR)/convergence/gaussian_0_05.png: \
		$(CONV_SCRIPT) $(CONV_AGENTS:%=$(TEST_DIR)/convergence/gaussian_0_05/%/progress.csv)
	python $^ --poly 3 5 --labels "RL" "WENO" --title "Convergence on Gaussian" \
		--output $(FIG_DIR)/convergence/gaussian_0_05.png
$(FIG_DIR)/convergence/gaussian_0_1.png: \
        	$(CONV_SCRIPT) $(CONV_AGENTS:%=$(TEST_DIR)/convergence/gaussian_0_1/%/progress.csv)
	python $^ --poly 3 --labels "RL" "WENO" --title "Convergence on Gaussian" \
		--output $(FIG_DIR)/convergence/gaussian_0_1.png

$(TEST_DIR)/convergence/smooth_sine_0_05/rl/progress.csv:
	$(RUN_CONV) --agent $(RL_AGENT) --init-type smooth_sine --time-max 0.05 --log-dir $(@D)
$(TEST_DIR)/convergence/smooth_sine_0_05/weno/progress.csv:
	$(RUN_CONV) --agent weno --init-type smooth_sine --time-max 0.05 --log-dir $(@D)
$(TEST_DIR)/convergence/smooth_sine_0_1/rl/progress.csv:
	$(RUN_CONV) --agent $(RL_AGENT) --init-type smooth_sine --time-max 0.1 --log-dir $(@D)
$(TEST_DIR)/convergence/smooth_sine_0_1/weno/progress.csv:
	$(RUN_CONV) --agent weno --init-type smooth_sine --time-max 0.1 --log-dir $(@D)

$(TEST_DIR)/convergence/gaussian_0_05/rl/progress.csv:
	$(RUN_CONV) --agent $(RL_AGENT) --init-type gaussian --C 0.5 --time-max 0.05 --log-dir $(@D)
$(TEST_DIR)/convergence/gaussian_0_05/weno/progress.csv:
	$(RUN_CONV) --agent weno --init-type gaussian --C 0.5 --time-max 0.05 --log-dir $(@D)
$(TEST_DIR)/convergence/gaussian_0_1/rl/progress.csv:
	$(RUN_CONV) --agent $(RL_AGENT) --init-type gaussian --C 0.5 --time-max 0.1 --log-dir $(@D)
$(TEST_DIR)/convergence/gaussian_0_1/weno/progress.csv:
	$(RUN_CONV) --agent weno --init-type gaussian --C 0.5 --time-max 0.1 --log-dir $(@D)


TRAINING_PLOTS=$(FIG_DIR)/training/loss.png $(FIG_DIR)/training/reward.png $(FIG_DIR)/training/l2.png
training: $(TRAINING_PLOTS)
TRAIN_PLOT_SCRIPT=scripts/combine_summary_plots.py

# Leaving off the training progress.csv as that should not change. If we're using a newly trained agent,
# modify TRAIN_DIR instead.
$(TRAINING_PLOTS): $(TRAIN_PLOT_SCRIPT)
	python $(TRAIN_PLOT_SCRIPT) $(TRAIN_DIR)/progress.csv \
		--std-only --output-dir $(FIG_DIR)/training
# Do this instead to plot average with confidence interval (if TRAIN_DIR contains a bunch of seeds):
#python $(TRAIN_PLOT_SCRIPT) --avg $(TRAIN_DIR)/*/progress.csv \
#--std-only --output-dir $(FIG_DIR)/training
