ITER=5

# dejong
python optimize_hparams_dalpha.py --env _dejong --num_trial 500 --num_epoch 500 --rl_device cpu

# ackley
python optimize_hparams_dalpha.py --env _ackley --num_trial 500 --num_epoch 500 --rl_device cpu

# cartpole
python optimize_hparams_dalpha.py --env cartpole_swing_up --num_trial 500 --num_epoch 500 --rl_device cpu

# ant
python optimize_hparams_dalpha.py --env ant --num_trial 500 --num_epoch 500 --rl_device cpu

# hopper
python optimize_hparams_dalpha.py --env hopper --num_trial 500 --num_epoch 500 --rl_device cpu