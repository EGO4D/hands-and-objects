"""
Default Configs
Refer: https://github.com/rbgirshick/yacs
"""

from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# -----------------------------------------------------------------------------
# Ego4D keystep localisation data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# Path to keystep localisation dataset
_C.DATA.VIDEO_DIR_PATH = "/ssd_scratch/cvit/sid/ego4d/videos"

# Path to keystep localisation annotations
_C.DATA.ANN_PATH = ("/home/siddhant.b/Ego4D_keyframe_localisation/annotations/"
                    "fho_miniset_v1.json")

# Path to keystep localisation splits
_C.DATA.SPLIT_PATH = ("/home/siddhant.b/Ego4D_keyframe_localisation/annotatio"
                      "ns/splits.json")

# Path to directory for temporarily or permanentaly storing clipped videos
_C.DATA.CLIPS_SAVE_PATH = "/ssd_scratch/cvit/sid/ego4d/temp_clips_folder"

# Path to directory containing no-state change videos
_C.DATA.NO_SC_PATH = ("/ssd_scratch/cvit/sid/ego4d_benchmark/data/no_state_"
                      "change_clips")

# Path to the no state change clips splits
_C.DATA.NO_SC_SPLIT_PATH = ("/home/sid/Ego4D_keyframe_localisation/"
                            "no_sc_splits.json")

# Path to the csv containing video component location information
_C.DATA.VIDEO_LOCATIONS_CSV = ("/mnt/nas/datasets/ego4d-miniset/consortium-sh"
                               "aring/dataset_manifests/video_component_loca"
                               "tions_university_video_access.csv")

# Rate at which we wish to sample the 8 seconds clips provided
_C.DATA.SAMPLING_FPS = 4

# Length of clips in seconds
_C.DATA.CLIP_LEN_SEC = 8

# Decoding backend, options include `pyav` or `torchvision`
_C.DATA.DECODING_BACKEND = "pyav"

# Input videos may has different fps, convert it to the target video fps before
# frame sampling.
_C.DATA.TARGET_FPS = 30

# List of input frame channel dimensions
_C.DATA.INPUT_CHANNEL_NUM = [3]

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 8

# The spatial crop size of the input clip.
_C.DATA.CROP_SIZE = 224

# Task to be performed
_C.DATA.TASK = "frame_detection_regression"

# If True, revert the default input channel (RBG <-> BGR).
_C.DATA.REVERSE_INPUT_CHANNEL = False

_C.DATA.FLOW_OUT_DIR = None

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]
# List of input frame channel dimensions.

_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

# ----------------------------------------------------------------------------
# Training options
# ----------------------------------------------------------------------------
_C.TRAIN = CfgNode()

# If True, train the model, else skip traning
_C.TRAIN.TRAIN_ENABLE = True

# Dataset
_C.TRAIN.DATASET = 'Ego4DKeyframeLocalisation'

# Batch size
_C.TRAIN.BATCH_SIZE = 8

# Batch size for validation
_C.TRAIN.VAL_BATCH_SIZE = 32

# Do one validation epoch only
_C.TRAIN.VAL_ONLY = False

# ----------------------------------------------------------------------------
# Testing options
# ----------------------------------------------------------------------------
_C.TEST = CfgNode()

# If true, test the model, else skip testing
_C.TEST.ENABLE = True

# Dataset for testing
_C.TEST.DATASET = 'Ego4DKeyframeLocalisation'

# Batch size
_C.TEST.BATCH_SIZE = 4

# Path to json file containing details about clipping of test and validation 
# clips. This is to make sure that we have the same test every time we run our
# code
_C.TEST.JSON = 'fixed_test_set.json'
_C.TEST.VAL_JSON = 'fixed_val_set.json'

# If False, adds final activation to model when evaluating
_C.TEST.NO_ACT = False  # True

# ----------------------------------------------------------------------------
# Common train/test data loader options
# ----------------------------------------------------------------------------
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory
_C.DATA_LOADER.PIN_MEMORY = True

# Shuffle the data
_C.DATA_LOADER.SHUFFLE = True

# Enable multi thread decoding
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False

# Option to use Hard or soft label for keyframe classification
_C.DATA_LOADER.SOFT_LABELS = False

# Drop the last batch
_C.DATA_LOADER.DROP_LAST = True

# If True, then load the non-state change clip's frames too
_C.DATA_LOADER.IS_NO_STATE_CHANGE = True

# -----------------------------------------------------------------------------
# Ego4D keystep localisation Misc options
# -----------------------------------------------------------------------------
_C.MISC = CfgNode()
# Option to see detailed outputs of steps taking place
_C.MISC.VERBOSE = True

# Path to save/pre-trained model
_C.MISC.CHECKPOINT_FILE_PATH = None

# Path to save outputs from lightning's Trainer
_C.MISC.OUTPUT_DIR = "/ssd_scratch/cvit/sid/ego4d/results"

# Number of GPUs to use
_C.MISC.NUM_GPUS = [0]

# Number of machines to use
_C.MISC.NUM_SHARDS = 1

# Whether to enable logging
_C.MISC.ENABLE_LOGGING = True

# Run 1 train, val, and test batch for debugging
_C.MISC.FAST_DEV_RUN = 1

# Run the whole training routine just on 50 samples (to check if the training
# code is working fine by overfitting the model)
_C.MISC.TEST_TRAIN_CODE = True

_C.MISC.LOG_FREQUENCY = 500
# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# BN epsilon.
_C.BN.EPSILON = 1e-5

_C.BN.FREEZE = False

# BN momentum.
_C.BN.MOMENTUM = 0.1

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for SplitBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.
_C.BN.NUM_SPLITS = 1

# Parameter for NaiveSyncBatchNorm3d, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized.
_C.BN.NUM_SYNC_DEVICES = 1

# -----------------------------------------------------------------------------
# ResNet options
# -----------------------------------------------------------------------------
_C.RESNET = CfgNode()

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = True

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
_C.MODEL.ARCH = "slow"

# Model name
_C.MODEL.MODEL_NAME = "ResNet"

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = [2]

# Number of classes to predict for state change head
_C.MODEL.NUM_STATE_CLASSES = [2]

# Reduction mode for calculating the loss
_C.MODEL.LOSS_REDUCTION = "none"

# Weight for keyframe detection loss
_C.MODEL.LAMBDA_1 = 1

# Weight for state change detection loss
_C.MODEL.LAMBDA_2 = 1

# The number of verbs to predict for the model
_C.MODEL.NUM_VERBS = 125

# The number of nouns to predict for the model
_C.MODEL.NUM_NOUNS = 352

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Verb loss function.
_C.MODEL.VERB_LOSS_FUNC = "cross_entropy"

# Next-Active-Object classification loss function.
_C.MODEL.NAO_LOSS_FUNC = "bce_logit"

# TTC loss function.
_C.MODEL.TTC_LOSS_FUNC = "smooth_l1"

# STA loss weights.
_C.MODEL.STA_LOSS_WEIGHTS = [1, 1, 1]  # VERB, NOUN, TTI

# Model architectures that has one single pathway.
_C.MODEL.SINGLE_PATHWAY_ARCH = ["c2d", "i3d", "slow"]

# Model architectures that has multiple pathways.
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"

# Activation layer for the output verb head.
_C.MODEL.HEAD_VERB_ACT = "softmax"

# Activation layer for the output tti head.
_C.MODEL.HEAD_TTC_ACT = "scaled_sigmoid"

# The maximum TTC value the model should predict
_C.MODEL.TTC_SCALE = 2

# Activation layer for the output noun head.
_C.MODEL.HEAD_NAO_ACT = "sigmoid"

# Activation layer for State change detection head
_C.MODEL.STATE_CHANGE_ACT = "softmax_2"

# Activation layer for keyframe detection head
_C.MODEL.KEYFRAME_DETECTION_ACT = "softmax_1"

_C.MODEL.CLASS_WEIGHTS = None

# -----------------------------------------------------------------------------
# Perceiver options
# -----------------------------------------------------------------------------

_C.PERCEIVER = CfgNode()
# number of channels for each token of the input
_C.PERCEIVER.INPUT_AXIS = 3

_C.PERCEIVER.NUM_SUBSETS = 2

# number of freq bands, with original value (2 * K + 1)
_C.PERCEIVER.NUM_FREQ_BANDS = 6

# maximum frequency, hyperparameter depending on how fine the data is
_C.PERCEIVER.MAX_FREQ = 10

# depth of net. The shape of the final attention mechanism will be:  depth * (cross attention -> self_per_cross_attn * self attention)
_C.PERCEIVER.DEPTH = 6

# number of latents, or induced set points, or centroids. different papers giving it different names
_C.PERCEIVER.NUM_LATENTS = 256

# latent dimension
_C.PERCEIVER.LATENT_DIM = 1024

# number of heads for cross attention. paper said 1
_C.PERCEIVER.CROSS_HEADS = 1

# number of heads for latent self attention, 8
_C.PERCEIVER.LATENT_HEADS = 8

# number of dimensions per cross attention head
_C.PERCEIVER.CROSS_DIM_HEAD = 64

# number of dimensions per latent self attention head
_C.PERCEIVER.LATENT_DIM_HEAD = 64

_C.PERCEIVER.ATTN_DROPOUT = 0.0
_C.PERCEIVER.FF_DROPOUT = 0.0

# whether to weight tie layers (optional, as indicated in the diagram)
_C.PERCEIVER.WEIGHT_TIE_LAYERS = False

# number of self attention blocks per cross attention
_C.PERCEIVER.SELF_PER_CROSS_ATTN = 6

# -----------------------------------------------------------------------------
# Nonlocal options
# -----------------------------------------------------------------------------
_C.NONLOCAL = CfgNode()

# Index of each stage and block to add nonlocal layers.
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

# Number of group for nonlocal for each stage.
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

# Instatiation to use for non-local layer.
_C.NONLOCAL.INSTANTIATION = "dot_product"

# Size of pooling layers used in Non-Local.
_C.NONLOCAL.POOL = [
    # Res2
    [1, 2, 2],
    # Res3
    [1, 2, 2],
    # Res4
    [1, 2, 2],
    # Res5
    [1, 2, 2],
]

# -----------------------------------------------------------------------------
# SlowFast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CfgNode()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
_C.SLOWFAST.ALPHA = 8

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
_C.SLOWFAST.FUSION_KERNEL_SZ = 5

_C.SLOWFAST.USE_PRETRAINED = False

_C.SLOWFAST.FREEZE_PRETRAINED = False

# -----------------------------------------------------------------------------
# Optimizer options
# -----------------------------------------------------------------------------
_C.SOLVER = CfgNode()

# Base learning rate
_C.SOLVER.BASE_LR = 0.1

# Number of epochs
_C.SOLVER.MAX_EPOCH = 100

# L2 regularization
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Optimization method
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Momentum
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum
_C.SOLVER.NESTEROV = True

# Which PyTorch Lightning accelerator to us
# Default dp (Data Parallel)
# https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html
_C.SOLVER.ACCELERATOR = "dp"

# Learning rate policy
_C.SOLVER.LR_POLICY = "cosine"

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Accumulate gradients
_C.SOLVER.ACCUMULATE_GRAD_BATCHES = None

_C.PROJECT = ""


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values"""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
