import os

class GlobalConfig:
    """ base architecture configurations """
	# Data
    seq_len = 1 # input timesteps
    pred_len = 4 # future waypoints predicted

    ignore_sides = True # don't consider side cameras
    ignore_rear = True # don't consider rear cameras
    n_views = 1 # no. of camera views

    input_resolution = 256

    scale = 1 # image pre-processing
    crop = 256 # image pre-processing

    num_class = 7
    classes = {
        0: [0, 0, 0],  # unlabeled
        1: [0, 0, 255],  # vehicle
        2: [128, 64, 128],  # road
        3: [255, 0, 0],  # red light
        4: [0, 255, 0],  # pedestrian
        5: [157, 234, 50],  # road line
        6: [255, 255, 255],  # sidewalk
    }
    converter = [
        0,  # unlabeled
        0,  # building
        0,  # fence
        0,  # other
        4,  # pedestrian
        0,  # pole
        5,  # road line
        2,  # road
        6,  # sidewalk
        0,  # vegetation
        1,  # vehicle
        0,  # wall
        0,  # traffic sign
        0,  # sky
        0,  # ground
        0,  # bridge
        0,  # rail track
        0,  # guard rail
        0,  # traffic light
        0,  # static
        0,  # dynamic
        0,  # water
        0,  # terrain
        3,  # red light
        3,  # yellow light
        0,  # green light
        0,  # stop sign
        5,  # stop line marking
    ]

    lr = 1e-4 # learning rate
    ls_seg   = 1.0
    ls_depth = 10.0

    # Conv Encoder
    vert_anchors = 8
    horz_anchors = 8
    anchors = vert_anchors * horz_anchors

	# GPT Encoder
    n_embd = 512
    block_exp = 4
    n_layer = 8
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    # Controller
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40 # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40 # buffer size

    max_throttle = 0.75 # upper limit on throttle signal value in dataset
    brake_speed = 0.1 # desired speed below which brake is triggered
    brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.25 # maximum change in speed input to logitudinal controller

    def __init__(self, machine=0, **kwargs):
        if (machine == 0):
            self.root_dir = '/mnt/qb/geiger/kchitta31/data_06_21'
            self.train_towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10']
            self.val_towns   = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06']
            self.train_data, self.val_data = [], []
            for town in self.train_towns:
                self.train_data.append(os.path.join(self.root_dir, town + ''))
                self.train_data.append(os.path.join(self.root_dir, town + '_small'))
            for town in self.val_towns:
                self.val_data.append(os.path.join(self.root_dir, town + '_long'))
        elif(machine == 1): #For local debug purposes
            self.root_dir = r"C:\Users\Admin\Ordnung\Studium\Master_Informatik\Masterarbeit\Masterthesis\src\DirectPerception\data"
            self.train_towns = ['Town01']
            self.val_towns = ['Town01']
            self.train_data, self.val_data = [], []
            for town in self.train_towns:
                self.train_data.append(os.path.join(self.root_dir, town + ''))
            for town in self.val_towns:
                self.val_data.append(os.path.join(self.root_dir, town + ''))
        else:
            self.root_dir = r"C:\Users\admin\Ordnung\Studium\Masterarbeit\Masterthesis\src\DirectPerception\data"
            self.train_towns = ['Town01']
            self.val_towns = ['Town01']
            self.train_data, self.val_data = [], []
            for town in self.train_towns:
                self.train_data.append(os.path.join(self.root_dir, town + ''))
            for town in self.val_towns:
                self.val_data.append(os.path.join(self.root_dir, town + ''))
        
        for k,v in kwargs.items():
            setattr(self, k, v)
