#!/bin/bash
trap "exit" INT

############################################################
# This is a template for creating a parameter sweep script. Make a copy of this file, don't edit it directly.
############################################################

###########################################################
# Declare your available parameters here.
###########################################################
init_types=("sine" "smooth_sine" "random" "rarefaction" "smooth_rare" "accelshock" "schedule" "sample")
eps_values=("0.01" "0.005" "0.003" "0.001")
order_values=("2" "3")

###########################################################
# Add each parameter list to the global list of parameters here.
###########################################################
declare -A param_values
param_values["--init-type"]=${init_types[@]}
param_values["--eps"]=${eps_values[@]}
param_values["--order"]=${order_values[@]}


params=(${!param_values[@]})
declare -A selected

function accumulate_params()
{
	local index=$1
	if [[ $index -eq ${#params[@]} ]]
	then
		###########################################################
		# Adjust the base python command as necessary.
		# Any parameters that should be the same for every run should go here.
		###########################################################
		cmd="python run_test.py -n"

		###########################################################
		# Some arguments, namely log-dir, may require more careful manipulation.
		# The current implementation creates directories for each parameter value.
		###########################################################
		log_dir="test/param_sweep"

		for param in ${params[@]}; do
			value=${selected["${param}"]}
			cmd="${cmd} ${param} ${value}"
			log_dir="${log_dir}/${value}"
		done

		cmd="${cmd} --log-dir ${log_dir}"
		echo $cmd

		###########################################################
		# Leave this line commented for a dry run to check that the commands look right,
		# then uncomment it to run.
		###########################################################
		#eval $cmd


	else
		local param=${params[$index]}
		local value_list=(${param_values["${param}"]})
		for value in ${value_list[@]}; do
			selected["${param}"]="${value}"

			next=$(($1+1))
			accumulate_params $next
		done
	fi
	return 0

}

accumulate_params 0
