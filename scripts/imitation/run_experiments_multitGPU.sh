# First, run the experiments. 3 multiagent models and 3 singleagent models.
# They can all be run in parallel. After they are done, we will generate the validation trajectories.
# There is also the fine tuning step in between.

CUDA_VISIBLE_DEVICES=0 python multiagent_curriculum_training.py --exp_name multiagent_curr_1_{} > logs/run_exp/init_run_full_0GPU.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python multiagent_curriculum_training.py --exp_name multiagent_curr_2_{} > logs/run_exp/init_run_full_1GPU.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python multiagent_curriculum_training.py --exp_name multiagent_curr_3_{} > logs/run_exp/init_run_full_2GPU.log 2>&1

CUDA_VISIBLE_DEVICES=0 python imitate.py --exp_name singleagent_def_1 --env_multiagent False --use_infogail False --policy_recurrent True --n_itr 1000 --n_envs 50 --validator_render False  --batch_size 10000 --gradient_penalty 2 --discount .95 --recurrent_hidden_dim 64 >> logs/run_exp/init_run_full_0GPU.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python imitate.py --exp_name singleagent_def_2 --env_multiagent False --use_infogail False --policy_recurrent True --n_itr 1000 --n_envs 50 --validator_render False  --batch_size 10000 --gradient_penalty 2 --discount .95 --recurrent_hidden_dim 64 >> logs/run_exp/init_run_full_1GPU.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python imitate.py --exp_name singleagent_def_3 --env_multiagent False --use_infogail False --policy_recurrent True --n_itr 1000 --n_envs 50 --validator_render False  --batch_size 10000 --gradient_penalty 2 --discount .95 --recurrent_hidden_dim 64  >> logs/run_exp/init_run_full_2GPU.log 2>&1

FAIL=0
for job in `jobs -p`
do
	wait $job || let "FAIL+=1"
done

echo $FAIL

# Now, FINE TUNE

CUDA_VISIBLE_DEVICES=0 python imitate.py --exp_name multiagent_curr_1_fine --env_multiagent True --use_infogail False --policy_recurrent True --n_itr 200 --n_envs 100 --validator_render False  --batch_size 40000 --gradient_penalty 2 --discount .99 --recurrent_hidden_dim 64 --params_filepath ../../data/experiments/multiagent_curr_1_50/imitate/log/itr_200.npz >> logs/run_exp/init_run_full_0GPU.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python imitate.py --exp_name multiagent_curr_2_fine --env_multiagent True --use_infogail False --policy_recurrent True --n_itr 200 --n_envs 100 --validator_render False  --batch_size 40000 --gradient_penalty 2 --discount .99 --recurrent_hidden_dim 64 --params_filepath ../../data/experiments/multiagent_curr_2_50/imitate/log/itr_200.npz >> logs/run_exp/init_run_full_1GPU.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python imitate.py --exp_name multiagent_curr_3_fine --env_multiagent True --use_infogail False --policy_recurrent True --n_itr 200 --n_envs 100 --validator_render False  --batch_size 40000 --gradient_penalty 2 --discount .99 --recurrent_hidden_dim 64 --params_filepath ../../data/experiments/multiagent_curr_3_50/imitate/log/itr_200.npz >> logs/run_exp/init_run_full_2GPU.log 2>&1



CUDA_VISIBLE_DEVICES=0 python imitate.py --exp_name singleagent_def_1_fine --env_multiagent False --use_infogail False --policy_recurrent True --n_itr 200 --n_envs 100 --validator_render False  --batch_size 40000 --gradient_penalty 2 --discount .99 --recurrent_hidden_dim 64 --params_filepath ../../data/experiments/singleagent_def_1/imitate/log/itr_1000.npz >> logs/run_exp/init_run_full_0GPU.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python imitate.py --exp_name singleagent_def_2_fine --env_multiagent False --use_infogail False --policy_recurrent True --n_itr 200 --n_envs 100 --validator_render False  --batch_size 40000 --gradient_penalty 2 --discount .99 --recurrent_hidden_dim 64 --params_filepath ../../data/experiments/singleagent_def_2/imitate/log/itr_1000.npz >> logs/run_exp/init_run_full_1GPU.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python imitate.py --exp_name singleagent_def_3_fine --env_multiagent False --use_infogail False --policy_recurrent True --n_itr 200 --n_envs 100 --validator_render False  --batch_size 40000 --gradient_penalty 2 --discount .99 --recurrent_hidden_dim 64 --params_filepath ../../data/experiments/singleagent_def_3/imitate/log/itr_1000.npz >> logs/run_exp/init_run_full_2GPU.log 2>&1

FAIL=0
for job in `jobs -p`
do
	wait $job || let "FAIL+=1"
done

echo $FAIL

# VALIDATE - creates the validation trajectories - simulates the model on each road section
for model in 1_fine 2_fine 3_fine; #run 2 at a time
do
    CUDA_VISIBLE_DEVICES=0 python validate.py --n_proc 20 --exp_dir ../../data/experiments/multiagent_curr_$model/ --params_filename itr_200.npz --use_multiagent True --random_seed 3 --n_envs 100 >> logs/run_exp/init_run_full_validate1.log 2>&1 &

    CUDA_VISIBLE_DEVICES=1 python validate.py --n_proc 20 --exp_dir ../../data/experiments/singleagent_def_$model/ --params_filename itr_200.npz --use_multiagent True --random_seed 3  --n_envs 100 >> logs/run_exp/init_run_full_validate2.log 2>&1

    for job in `jobs -p`
    do
        echo $job
        wait $job || let "FAIL+=1"
    done
done

#Now that validation is done, there will be .npz trajectory files for each of the experiments
# These should appear in ../../data/experiments/{model_name}/imiate/validation/

# Now, you can run visualize.py, or use the visualize ipython notebooks (recommended) to examine the results.

