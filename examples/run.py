import sys
from pathlib import Path
from hloc import extract_features, match_features


BASE_DIR = sys.argv[1]
BASE_DIR = Path(BASE_DIR)

path_images = BASE_DIR / 'images'
path_keypoints = BASE_DIR / 'keypoints.h5'

keypoints_conf = extract_features.confs["r2d2"]
matcher_conf = match_features.confs["superglue"]

# extract sparse local features for mapping and query images
extract_features.main(keypoints_conf, path_images, feature_path=path_keypoints)
