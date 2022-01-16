# Makefile for the figures needed for the paper.
# This file requires that the RL_AGENT defined below already exists.
# This file does not compile source code in anyway.

SHELL=/bin/bash

ANALYTICAL_EXCEPT_SMOOTH_SINE=accelshock gaussian smooth_rare rarefaction shock tophat other_sine
ANALYTICAL_INITS=smooth_sine $(ANALYTICAL_EXCEPT_SMOOTH_SINE)
OTHER_INITS=sine line para sawtooth 
ALL_INITS=$(ANALYTICAL_INITS) $(OTHER_INITS)
EVAL_INITS=smooth_sine rarefaction accelshock other_sine gaussian tophat

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
#RL_AGENT=log/weno_burgers/full/raresched_sweep/seed_1/model_best.zip
#TRAIN_DIR=log/weno_burgers/full/raresched_sweep/*
RL_AGENT=log/weno_burgers/sweep_hyper3/layers_64_64/learning-rate_0_0003/seed_3/model_best.zip
TRAIN_DIR=log/weno_burgers/sweep_hyper3/layers_64_64/learning-rate_0_0003/*


RUN_TEST=python run_test.py -y --animate --output-mode csv plot $\
		--plot-tv --plot-l2 --plot-error --plot-actions --evolution-plot $\
		--rk rk4 --fixed-timesteps $\
		--order $(ORDER)

all: plots
plots: state convergence tv l2 action animation training error_comparison

state: $(ALL_INITS:%=%_state)

STATE_SHORTCUTS=$(ALL_INITS:%=%_state)
$(STATE_SHORTCUTS): %_state: $(FIG_DIR)/states/%.png

END_STATE=burgers_state_step500.csv
HALF_STATE=burgers_state_step250.csv
INIT_STATE=burgers_state_step000.csv

ALL_AGENTS=analytical weno rl
#A_AGENTS=analytical rl
A_AGENTS=analytical weno rl
O_AGENTS=weno rl

# The original state plot was only the final state and used these rules:
#STATE_SCRIPT=scripts/combine_state_plots.py

#$(FIG_DIR)/states/smooth_sine.png: $(STATE_SCRIPT) $(ALL_AGENTS:%=$(TEST_DIR)/smooth_sine/%/$(END_STATE))
	#python $^ --labels "true solution" "WENO solution" "RL solution" \
		#--title "RL solution at t=0.2" --output $@
#
#define A_STATE_RULE
#$$(FIG_DIR)/states/$(init).png: $$(STATE_SCRIPT) $$(A_AGENTS:%=$$(TEST_DIR)/$(init)/%/$$(END_STATE))
	#python $$^ --labels "true solution" "RL solution" \
		#--title "RL solution at t=0.2" --output $$(FIG_DIR)/states/$(init).png
#endef
#$(foreach init,$(ANALYTICAL_EXCEPT_SMOOTH_SINE), $(eval $(A_STATE_RULE)))
#
#define O_STATE_RULE
#$$(FIG_DIR)/states/$(init).png: $$(STATE_SCRIPT) $$(O_AGENTS:%=$$(TEST_DIR)/$(init)/%/$$(END_STATE))
	#python $$^ --labels "WENO solution" "RL solution" \
		#--title "RL solution at t=0.2" --output $(FIG_DIR)/states/$(init).png
#endef
#$(foreach init,$(OTHER_INITS), $(eval $(O_STATE_RULE)))

STATE_SCRIPT=scripts/half_evolution_plot.py

$(FIG_DIR)/states/smooth_sine.png: $(STATE_SCRIPT) $(ALL_AGENTS:%=$(TEST_DIR)/smooth_sine/%/$(END_STATE))
	python $< --init $(TEST_DIR)/smooth_sine/rl/$(INIT_STATE) \
		--rl $(TEST_DIR)/smooth_sine/rl/$(HALF_STATE) \
			$(TEST_DIR)/smooth_sine/rl/$(END_STATE) \
		--weno $(TEST_DIR)/smooth_sine/weno/$(HALF_STATE) \
			$(TEST_DIR)/smooth_sine/weno/$(END_STATE) \
		--true $(TEST_DIR)/smooth_sine/analytical/$(HALF_STATE) \
			$(TEST_DIR)/smooth_sine/analytical/$(END_STATE) \
		--output $@

define A_STATE_RULE
$$(FIG_DIR)/states/$(init).png: $$(STATE_SCRIPT) $$(A_AGENTS:%=$$(TEST_DIR)/$(init)/%/$$(END_STATE))
	python $$< --init $$(TEST_DIR)/$(init)/rl/$$(INIT_STATE) \
		--rl $$(TEST_DIR)/$(init)/rl/$$(HALF_STATE) $$(TEST_DIR)/$(init)/rl/$$(END_STATE) \
		--weno $$(TEST_DIR)/$(init)/weno/$$(HALF_STATE) $$(TEST_DIR)/$(init)/weno/$$(END_STATE) \
		--true $$(TEST_DIR)/$(init)/analytical/$$(HALF_STATE) \
			$$(TEST_DIR)/$(init)/analytical/$$(END_STATE) \
		--no-legend --output $$@
endef
$(foreach init,$(ANALYTICAL_EXCEPT_SMOOTH_SINE), $(eval $(A_STATE_RULE)))

define O_STATE_RULE
$$(FIG_DIR)/states/$(init).png: $$(STATE_SCRIPT) $$(O_AGENTS:%=$$(TEST_DIR)/$(init)/%/$$(END_STATE))
	python $$< --init $$(TEST_DIR)/$(init)/rl/$$(INIT_STATE) \
		--rl $$(TEST_DIR)/$(init)/rl/$$(HALF_STATE) $$(TEST_DIR)/$(init)/rl/$$(END_STATE) \
		--weno $$(TEST_DIR)/$(init)/weno/$$(HALF_STATE) $$(TEST_DIR)/$(init)/weno/$$(END_STATE) \
		--no-legend --output $$@
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

# All files for a given experiment are considered to depend on the end state files.
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

l2: $(ANALYTICAL_INITS:%=%_l2) average_l2

L2_SHORTCUTS=$(ANALYTICAL_INITS:%=%_l2)
$(L2_SHORTCUTS): %_l2: $(FIG_DIR)/l2/%_l2.png

L2_SCRIPT=scripts/combine_time_plots.py
L2_AGENTS=rl weno

WENO_L2_FILES=$(EVAL_INITS:%=$(TEST_DIR)/%/weno/progress.csv)
RL_L2_FILES=$(EVAL_INITS:%=$(TEST_DIR)/%/rl/progress.csv)
average_l2: $(FIG_DIR)/l2/average_l2.png
$(FIG_DIR)/l2/average_l2.png: $(L2_SCRIPT) $(WENO_L2_FILES) $(RL_L2_FILES)
	python $< --avg $(WENO_L2_FILES) --avg $(RL_L2_FILES) --labels "WENO" "RL" \
		--ycol l2 --yscale log --ylabel "L2 Error" --ci-type none --output $@

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


conv: convergence
convergence: conv_short conv_long
conv_short: smooth_sine_conv gaussian_conv

CONV_SCRIPT=scripts/combine_convergence_plots.py
CONV_AGENTS=rl weno

RUN_CONV=python run_test.py --convergence-plot -y --order $(ORDER) --rk rk4 --variable-timesteps

smooth_sine_conv: $(FIG_DIR)/convergence/smooth_sine_0_05.png $(FIG_DIR)/convergence/smooth_sine_0_1.png
$(FIG_DIR)/convergence/smooth_sine_0_05.png: \
       		$(CONV_SCRIPT) $(CONV_AGENTS:%=$(TEST_DIR)/convergence/smooth_sine_0_05/%/progress.csv)
	python $^ --poly 3 5 --labels "RL" "WENO" --title "Convergence on sine" \
		--output $@
$(FIG_DIR)/convergence/smooth_sine_0_1.png: \
		$(CONV_SCRIPT) $(CONV_AGENTS:%=$(TEST_DIR)/convergence/smooth_sine_0_1/%/progress.csv)
	python $^ --labels "RL" "WENO" --title "Convergence on sine" \
		--output $@

gaussian_conv: $(FIG_DIR)/convergence/gaussian_0_05.png $(FIG_DIR)/convergence/gaussian_0_1.png
$(FIG_DIR)/convergence/gaussian_0_05.png: \
		$(CONV_SCRIPT) $(CONV_AGENTS:%=$(TEST_DIR)/convergence/gaussian_0_05/%/progress.csv)
	python $^ --poly 3 5 --labels "RL" "WENO" --title "Convergence on Gaussian" \
		--output $@
$(FIG_DIR)/convergence/gaussian_0_1.png: \
        	$(CONV_SCRIPT) $(CONV_AGENTS:%=$(TEST_DIR)/convergence/gaussian_0_1/%/progress.csv)
	python $^ --poly 3 --labels "RL" "WENO" --title "Convergence on Gaussian" \
		--output $@

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

conv_long: $(FIG_DIR)/convergence/average.png

CONV_INITS=$(EVAL_INITS)
WENO_CONV_FILES=$(CONV_INITS:%=$(TEST_DIR)/convergence/%/weno/progress.csv)
RL_CONV_FILES=$(CONV_INITS:%=$(TEST_DIR)/convergence/%/rl/progress.csv)

$(FIG_DIR)/convergence/average.png: $(CONV_SCRIPT) $(WENO_CONV_FILES) $(RL_CONV_FILES)
	python $< --avg $(WENO_CONV_FILES) --avg $(RL_CONV_FILES) --labels "WENO" "RL" \
		--title "Average Error Convergence" --ci-type none --output $@

$(TEST_DIR)/convergence/%/weno/progress.csv:
	$(RUN_CONV) --agent weno --init-type $* --time-max 0.2 --log-dir $(@D)
$(TEST_DIR)/convergence/%/rl/progress.csv:
	$(RUN_CONV) --agent $(RL_AGENT) --init-type $* --time-max 0.2 --log-dir $(@D)


TRAINING_PLOTS=$(FIG_DIR)/training/loss.png $(FIG_DIR)/training/reward.png $(FIG_DIR)/training/l2.png
training: $(TRAINING_PLOTS)
TRAIN_PLOT_SCRIPT=scripts/combine_summary_plots.py

# Leaving off the training progress.csv as that should not change. If we're using a newly trained agent,
# modify TRAIN_DIR instead.
$(TRAINING_PLOTS): $(TRAIN_PLOT_SCRIPT)
	python $(TRAIN_PLOT_SCRIPT) --avg $(TRAIN_DIR)/progress.csv \
		--std-only --output-dir $(FIG_DIR)/training


ERROR_COMPARISON_PLOT=$(FIG_DIR)/error_comparison/comparison.png
error_comparison: $(ERROR_COMPARISON_PLOT)

ERROR_COMPARISON_SCRIPT=scripts/error_comparison_plot.py
RANDOM_ENVS=random accelshock_random smooth_rare_random
NUM_SEEDS=400
SEEDS=$(shell seq 1 $(NUM_SEEDS))
RL_ERROR_FILES=$(foreach env,$(RANDOM_ENVS),\
			$(foreach seed,$(SEEDS),\
				$(TEST_DIR)/error_comparison/rl/$(env)/seed_$(seed)/progress.csv))
WENO_ERROR_FILES=$(foreach env,$(RANDOM_ENVS),\
			$(foreach seed,$(SEEDS),\
				$(TEST_DIR)/error_comparison/weno/$(env)/seed_$(seed)/progress.csv))

$(ERROR_COMPARISON_PLOT): $(ERROR_COMPARISON_SCRIPT) $(RL_ERROR_FILES) $(WENO_ERROR_FILES)
	python $< \
		--xname WENO --x-error $(TEST_DIR)/error_comparison/weno/*/seed_{1..$(NUM_SEEDS)}/progress.csv \
		--yname RL --y-error $(TEST_DIR)/error_comparison/rl/*/seed_{1..$(NUM_SEEDS)}/progress.csv \
		--output $@

RUN_RANDOM_TEST=python run_test.py -y --analytical --output-mode csv --plot-l2 \
		--order 3 --init-params random=cont --num-cells random \
		--rk 4 --time-max 0.2 --variable-timesteps

define RANDOM_WENO_RULE
$$(TEST_DIR)/error_comparison/weno/$(env)/seed_$(seed)/progress.csv:
	echo $$@;\
	$$(RUN_RANDOM_TEST) --agent weno --init-type $(env) --seed $(seed) --log-dir $$(@D)
endef
$(foreach env,$(RANDOM_ENVS), \
	$(foreach seed,$(SEEDS), \
		$(eval $(RANDOM_WENO_RULE))))
define RANDOM_RL_RULE
$$(TEST_DIR)/error_comparison/rl/$(env)/seed_$(seed)/progress.csv:
	echo $$@;\
	$$(RUN_RANDOM_TEST) --agent $$(RL_AGENT) --init-type $(env) --seed $(seed) --log-dir $$(@D)
endef
$(foreach env,$(RANDOM_ENVS), \
	$(foreach seed,$(SEEDS), \
		$(eval $(RANDOM_RL_RULE))))
