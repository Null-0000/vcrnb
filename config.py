# deeeeeebug
import os
USE_IMAGENET_PRETRAINED = True # otherwise use detectron, but that doesnt seem to work?!?

# Change these to match where your annotations and images are
USERNAME = 'wangkejie'
DATA_DIR = f'/home/share/{USERNAME}/vcr1'
# DATA_DIR = '/home/vcr1'
SAVE_DIR = '/home/share/wangkejie/SDUVCR/models/saves'
# SAVE_DIR = '/mnt/saves'
VCR_IMAGES_DIR = os.path.join(DATA_DIR, 'vcr1images')
VCR_ANNOTS_DIR = DATA_DIR
IMAGE_DESIRED_WIDTH = 768
IMAGE_DESIRED_HEIGHT = 384

if not os.path.exists(VCR_IMAGES_DIR):
    print(os.path.join(os.path.dirname(__file__), 'data', 'vcr1images'))
    raise ValueError("Update config.py with where you saved VCR images to.")


