# Command for running the test script on a saved model
command = 'python run.py --cfg configs/keyframe_localisation_train_only_13-05-21.yaml MISC.NUM_GPUS 2 MISC.CHECKPOINT_FILE_PATH ~/benchmark/train_only_13-05-21/lightning_logs/version_9/checkpoints/epoch\=24-step\=159261.ckpt TRAIN.TRAIN_ENABLE False TEST.ENABLE True MISC.NUM_GPUS 1 DATA_LOADER.NUM_WORKERS 8'
# Command being used to restart the training from a previously saved checkpoint
command = 'python run.py --cfg configs/keyframe_localisation_train_only_13-05-21.yaml MISC.NUM_GPUS 2 MISC.CHECKPOINT_FILE_PATH ~/benchmark/train_only_13-05-21/lightning_logs/version_9/checkpoints/epoch\=24-step\=159261.ckpt'
# Command to start a fresh training
command = 'python run.py --cfg configs/keyframe_localisation_train_only_13-05-21.yaml MISC.NUM_GPUS 2'
