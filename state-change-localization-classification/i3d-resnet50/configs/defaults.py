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

# Path to the directory containing annotation json files
_C.DATA.ANN_DIR = ("/home/sid/canonical_dataset/"
                    "competition-json-files_2021-11-13")

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

# Rate at which we wish to sample the 8 seconds clips provided
_C.DATA.SAMPLING_FPS = 4

# Length of clips in seconds
_C.DATA.CLIP_LEN_SEC = 8

# List of input frame channel dimensions
_C.DATA.INPUT_CHANNEL_NUM = [3]

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 8

# The spatial crop size of the input clip.
_C.DATA.CROP_SIZE = 224

# Task to be performed
_C.DATA.TASK = "frame_detection_regression"

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

# ----------------------------------------------------------------------------
# Testing options
# ----------------------------------------------------------------------------
_C.TEST = CfgNode()

# If true, test the model, else skip testing
_C.TEST.ENABLE = False

# Dataset for testing
_C.TEST.DATASET = 'Ego4DKeyframeLocalisation'

# Batch size
_C.TEST.BATCH_SIZE = 4

# Path to json file containing details about clipping of test and validation 
# clips. This is to make sure that we have the same test every time we run our
# code
_C.TEST.JSON = 'fixed_test_set.json'
_C.TEST.VAL_JSON = 'fixed_val_set.json'

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

# Drop the last batch
_C.DATA_LOADER.DROP_LAST = True

# If True, then load the non-state change clip's frames too
_C.DATA_LOADER.IS_NO_STATE_CHANGE = True

# -----------------------------------------------------------------------------
# Ego4D keystep localisation Misc options
# -----------------------------------------------------------------------------
_C.MISC = CfgNode()

# Path to save/pre-trained model
_C.MISC.CHECKPOINT_FILE_PATH = None

# Path to save outputs from lightning's Trainer
_C.MISC.OUTPUT_DIR = "/ssd_scratch/cvit/sid/ego4d/results"

# Number of GPUs to use
_C.MISC.NUM_GPUS = 1

# Number of machines to use
_C.MISC.NUM_SHARDS = 1

# Whether to enable logging
_C.MISC.ENABLE_LOGGING = True

# Run 1 train, val, and test batch for debugging
_C.MISC.FAST_DEV_RUN = False

# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

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
_C.MODEL.ARCH = "slow_layer5"

# Model name
_C.MODEL.MODEL_NAME = "DualHeadResNet"

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = [2]

# Number of classes to predict for state change head
_C.MODEL.NUM_STATE_CLASSES = [2]

# Reduction mode for calculating the loss
_C.MODEL.LOSS_REDUCTION = "none"

# Weight for keyframe localization loss
_C.MODEL.LAMBDA_1 = 1

# Weight for state change detection loss
_C.MODEL.LAMBDA_2 = 1

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"

# Activation layer for State change detection head
_C.MODEL.STATE_CHANGE_ACT = "softmax_2"

# Activation layer for keyframe detection head
_C.MODEL.KEYFRAME_DETECTION_ACT = "softmax_1"

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

# -----------------------------------------------------------------------------
# State change detection and keyframe localisation JSON export options
# -----------------------------------------------------------------------------
_C.JSON_EXPORT = CfgNode()

# If true, create test_annotated.json (containing the annotations for the test
# splits), else create test_unannotated.json (does not contain the annotations
# for the test split)
_C.JSON_EXPORT.TEST_ANNOTATED = True

# If true, check the generated text files to make sure clips striclty follow
# train, test, and validation splits
_C.JSON_EXPORT.CHECK_SPLITS = False

# Path to the directory for saving the json files
_C.JSON_EXPORT.SAVE_DIR = ('/home/sid/canonical_dataset/'
                            'competition-json-files_2021-11-13')

# State change mapping jsons path
_C.JSON_EXPORT.SC_MAPPING = ('/home/sid/canonical_dataset/state-change-id-to-'
                                'canonical-video-id-mapping-{}-2021-10-31.json')

# Mode in which to create the json files. Options are: ['train', 'test', 'val']
_C.JSON_EXPORT.MODE = 'train'

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values"""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
