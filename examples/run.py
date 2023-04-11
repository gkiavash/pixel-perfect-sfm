import sys
from pathlib import Path
from hloc import (
    extract_features,
    match_features,
    pairs_from_exhaustive
)


BASE_DIR = sys.argv[1]
BASE_DIR = Path(BASE_DIR)

path_images = BASE_DIR / 'images'
path_keypoints = BASE_DIR / 'keypoints.h5'
path_pairs = BASE_DIR / 'pairs.txt'

keypoints_conf = extract_features.confs["r2d2"]
matcher_conf = match_features.confs["superglue"]

extract_features.main(
    keypoints_conf,
    path_images,
    feature_path=path_keypoints
)

pairs_from_exhaustive.main(
    output=path_pairs,
    features=path_keypoints,
)
