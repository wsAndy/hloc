from pathlib import Path
import os
from hloc import extract_features, match_features, reconstruction, pairs_from_retrieval
import pycolmap
import logging
import shutil
from argparse import ArgumentParser


parser = ArgumentParser("HLOC SFM")
parser.add_argument("--images", "-i", required=True, type=str)
parser.add_argument("--output", "-o", default="", type=str)
parser.add_argument("--camera", default="PINHOLE", type=str)
parser.add_argument("--cameraMode", default="SINGLE", type=str)
args = parser.parse_args()

images = Path( args.images ) # Path( '/root/data/images/')

if not images.exists():
    logging.error(f" Images Folder not exists.")
    exit(-1)

outputs = Path(args.output)
if outputs == Path(''):
    outputs = Path(images, '..', 'sfm/')

sfm_pairs = outputs / 'pairs-netvlad.txt'
sfm_dir = outputs /  'colmap' # 'sfm_superpoint+superglue'

# 描述是否所有图片内参一样
# AUTO、SINGLE、PER_FOLDER、PER_IMAGE
if args.cameraMode == 'AUTO':
    camera_mode =  pycolmap.CameraMode.AUTO
elif args.cameraMode == 'SINGLE':
    camera_mode =  pycolmap.CameraMode.SINGLE
elif args.cameraMode == 'PER_FOLDER':
    camera_mode =  pycolmap.CameraMode.PER_FOLDER
elif args.cameraMode == 'PER_IMAGE':
    camera_mode =  pycolmap.CameraMode.PER_IMAGE

image_options = {'camera_model':  args.camera }

if image_options['camera_model'] != 'PINHOLE' and image_options['camera_model'] != 'SIMPLE_PINHOLE':
    distorter = True
if distorter:
    sfm_dir = sfm_dir / 'distorted' 

retrieval_conf = extract_features.confs['netvlad']
feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']

retrieval_path = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)

feature_path = extract_features.main(feature_conf, images, outputs)
match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)


model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path, camera_mode, False, False, None, None, image_options)

if distorter:
    '''
    如果走去畸变，那么最终结果存储在 E:\docker\nerfData\hloc\sfm\colmap 中，
    其中E:\docker\nerfData\hloc\sfm\colmap\distorted\ 里面是上述重建的结果
    所以，所有的最终结果都会存储在sfm_dir中
    '''
    img_undist_cmd = ("colmap image_undistorter \
        --image_path " + str(images) + " \
        --input_path " + str(sfm_dir) + "/distorted/sparse/0 \
        --output_path " + str(sfm_dir) + "\
        --output_type COLMAP --max_image_size 4096 ")

    exit_code = os.system(img_undist_cmd)

    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    files = os.listdir( sfm_dir / "sparse")
    os.makedirs( sfm_dir / "sparse/0", exist_ok=True)
    # Copy each file from the source directory to the destination directory
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join( sfm_dir, "sparse", file)
        destination_file = os.path.join( sfm_dir, "sparse", "0", file)
        shutil.move(source_file, destination_file)

else:
    '''
    拷贝原始图片到目标位置
    '''
    shutil.copytree(images, sfm_dir / 'images')

