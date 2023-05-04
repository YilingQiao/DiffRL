NUM_TRIALS=10

# dejong
python optimize_hparams_dalpha.py --env _dejong --num_trial ${NUM_TRIALS} --num_epoch -1

# ackley
python optimize_hparams_dalpha.py --env _ackley --num_trial ${NUM_TRIALS} --num_epoch -1

# cartpole
python optimize_hparams_dalpha.py --env cartpole_swing_up --num_trial ${NUM_TRIALS} --num_epoch -1

# ant
python optimize_hparams_dalpha.py --env ant --num_trial ${NUM_TRIALS} --num_epoch -1

# hopper
python optimize_hparams_dalpha.py --env hopper --num_trial ${NUM_TRIALS} --num_epoch -1

# cheetah
python optimize_hparams_dalpha.py --env cheetah --num_trial ${NUM_TRIALS} --num_epoch -1

# cheetah
python optimize_hparams_dalpha.py --env humanoid --num_trial ${NUM_TRIALS} --num_epoch -1

# cheetah
python optimize_hparams_dalpha.py --env snu_humanoid --num_trial ${NUM_TRIALS} --num_epoch -1