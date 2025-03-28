import os
class EnvironmentSettings:
    def __init__(self):
        project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

        # project path
        self.workspace_dir = project_path    # Base directory for saving network checkpoints.
        self.tensorboard_dir = os.path.join(project_path, 'tensorboard')  # Directory for tensorboard files.
        self.pretrained_networks = os.path.join(project_path, 'pretrained_networks')  

        # dataset path
        self.lasot_dir = '/home/lhg/work/ssd/LaSOT/github/LaSOTBenchmark/'
        self.got10k_dir = '/home/lhg/work/ssd/github_got10k/got10k'
        self.trackingnet_dir = '/home/lhg/work/ssd/github_trackingnet/trackingnet/'
        self.coco_dir = '/home/lhg/work/ssd/coco/github/results/'