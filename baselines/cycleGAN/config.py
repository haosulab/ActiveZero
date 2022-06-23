from yacs.config import CfgNode as CN

_C = CN()
_C.NAME = "TESTING PURPOSES"
cfg = _C

_C.MODEL = CN()
_C.MODEL.MAX_DISP = 192  # maximum disparity
_C.MODEL.BACKBONE = "psmnet" # psmnet, dispnet, raft
_C.MODEL.CROP_HEIGHT = 256  # crop height
_C.MODEL.CROP_WIDTH = 512  # crop width
_C.MODEL.ADAPTER = True
_C.MODEL.CHECKPOINT = "./model_best.pth"

# under the assumption that real and sim are trained seperately
_C.LOSSES = CN()
_C.LOSSES.SIMRATIO = 1.0
_C.LOSSES.REALRATIO = 1.0
_C.LOSSES.ONREAL = True
_C.LOSSES.ONSIM = True
_C.LOSSES.EXCLUDE_BG = True
_C.LOSSES.EXCLUDE_ZEROS = True

_C.LOSSES.DISP_LOSS = True # train on sim disparity
_C.LOSSES.REPROJECTION_LOSS = True # train on sim disparity

_C.LOSSES.REPROJECTION = CN()
_C.LOSSES.REPROJECTION.PATTERN = "temporal" # p1, p2, temporal, lcn, img
_C.LOSSES.REPROJECTION.PATCH_SIZE = 11
_C.LOSSES.REPROJECTION.TRAINREAL = True
_C.LOSSES.REPROJECTION.TRAINSIM = True
_C.LOSSES.REPROJECTION.REALRATIO = 1.0
_C.LOSSES.REPROJECTION.SIMRATIO = 1.0

##################### future losses also add here ###########################

# Split files
_C.SIM = CN()
_C.SIM.DATASET = "/cephfs/datasets/iccv_pnp/messy-table-dataset/v12/training/"  #  directory of your training dataset
_C.SIM.TRAIN = "/cephfs/datasets/iccv_pnp/messy-table-dataset/v12/training_lists/all_train.txt"  # training lists of your training dataset
_C.SIM.VAL = "/cephfs/datasets/iccv_pnp/messy-table-dataset/v12/training_lists/all_val.txt"  # training lists of your validation dataset
_C.SIM.TESTSET = '/cephfs/datasets/iccv_pnp/messy-table-dataset/real_v9/training/'
_C.SIM.TEST = "/cephfs/datasets/iccv_pnp/messy-table-dataset/real_v9/training_lists/all.txt"
_C.SIM.OBJ_NUM = 18  # Note: table + ground - 17th
_C.SIM.LEFT = "0128_irL_kuafu_half.png" # "0128_irL_kuafuv2_half.png on testing
_C.SIM.LEFT_NO_IR = "0128_irL_kuafu_half_no_ir.png"
_C.SIM.RIGHT = "0128_irR_kuafu_half.png"
_C.SIM.RIGHT_NO_IR = "0128_irR_kuafu_half_no_ir.png"
_C.SIM.DEPTH = "depth.png"
_C.SIM.DEPTHL = "depthL.png"
_C.SIM.DEPTHR = "depthR.png"
_C.SIM.META = "meta.pkl"
_C.SIM.LABEL = "irL_label_image.png"
_C.SIM.REALSENSE = "0128_depth_denoised.png"

_C.REAL = CN()
_C.REAL.TEST = '/cephfs/datasets/iccv_pnp/messy-table-dataset/real_v9/training_lists/all.txt'
_C.REAL.TRAIN = "/cephfs/datasets/iccv_pnp/messy-table-dataset/rand_scenes/train_list.txt"
_C.REAL.LABELSET = "/cephfs/datasets/iccv_pnp/real_data_v9/"
_C.REAL.TESTSET =  '/cephfs/datasets/iccv_pnp/real_data_v9/'
_C.REAL.DATASET = "/cephfs/datasets/iccv_pnp/messy-table-dataset/rand_scenes/rand_scenes/"  # path to your real testing dataset
_C.REAL.LEFT = "1024_irL_real_360.png"
_C.REAL.LEFT_NO_IR = "1024_irL_real_off.png"
_C.REAL.LEFT_TEMPORAL_IR = "1024_irL_real_temporal.png"
_C.REAL.RIGHT = "1024_irR_real_360.png"
_C.REAL.RIGHT_NO_IR = "1024_irR_real_off.png"
_C.REAL.RIGHT_TEMPORAL_IR = "1024_irR_real_temporal.png"
_C.REAL.PAD_WIDTH = 960
_C.REAL.PAD_HEIGHT = 544
_C.REAL.MASK_FILE = "/isabella-fast/FeatureGAN/real_masks/all.txt"
_C.REAL.MASK = "/isabella-fast/FeatureGAN/real_masks"
_C.REAL.REALSENSE = "1024_depth_real.png"
_C.REAL.OBJ = [4, 5, 7, 9, 13, 14, 15, 16]

_C.SOLVER = CN()
_C.SOLVER.LR = 0.0002  # base learning rate for cascade
_C.SOLVER.LR_G = 0.0002
_C.SOLVER.LR_D = 0.0002
_C.SOLVER.LR_GAN_STEPS = '5000, 10000, 15000:10'
_C.SOLVER.LR_STEPS = '10000,20000,30000,40000:2'  # the steps to decay lr: the downscale rate
_C.SOLVER.BETAS = (0.9, 0.999)
_C.SOLVER.EPOCHS = 20  # number of epochs to train
_C.SOLVER.STEPS = 50000  # number of steps to train
_C.SOLVER.BATCH_SIZE = 1  # batch size
_C.SOLVER.NUM_WORKER = 1  # num_worker in dataloader
_C.SOLVER.DEBUG = False
_C.SOLVER.SUB = 100
_C.SOLVER.LOGDIR = "./"
_C.SOLVER.SAVE_FREQ= 1000
_C.SOLVER.SUMMARY_FREQ = 500
_C.SOLVER.SEED = 1

# Data Augmentation
_C.DATA_AUG = CN()
_C.DATA_AUG.COLOR_JITTER = True
_C.DATA_AUG.GAUSSIAN_BLUR = True
_C.DATA_AUG.BRIGHT_MIN = 0.4
_C.DATA_AUG.BRIGHT_MAX = 1.4
_C.DATA_AUG.CONTRAST_MIN = 0.8
_C.DATA_AUG.CONTRAST_MAX = 1.2
_C.DATA_AUG.GAUSSIAN_MIN = 0.1
_C.DATA_AUG.GAUSSIAN_MAX = 2.0
_C.DATA_AUG.GAUSSIAN_KERNEL = 9
