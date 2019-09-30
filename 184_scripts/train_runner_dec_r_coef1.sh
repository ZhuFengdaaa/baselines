export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zhufengda/.mujoco/mujoco200/bin
CUDA_VISIBLE_DEVICES=0 python -m baselines.run --alg=cppo2 --env=CAntMaze-v1 --num_timesteps=1e6 --save_path ~/models/dec_r_coef1 --save_interval 10 --num_env 8
