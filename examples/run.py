import sys
from pathlib import Path
from hloc import (
    extract_features,
    match_features,
    pairs_from_exhaustive
)
from pixsfm.refine_hloc import PixSfM


BASE_DIR = sys.argv[1]
BASE_DIR = Path(BASE_DIR)

path_images = BASE_DIR / 'images'
path_keypoints = BASE_DIR / 'keypoints.h5'
path_pairs = BASE_DIR / 'pairs.txt'
path_matches = BASE_DIR / 'matches.h5'
path_sfm = BASE_DIR / 'sfm/'
path_cache = BASE_DIR / f'dense_features_cache.h5'

keypoints_conf = extract_features.confs["r2d2"]
matcher_conf = match_features.confs["NN-superpoint"]

extract_features.main(
    keypoints_conf,
    path_images,
    as_half=False,
    feature_path=path_keypoints
)

pairs_from_exhaustive.main(
    output=path_pairs,
    features=path_keypoints,
)

match_features.main(
    conf=matcher_conf,
    pairs=path_pairs,
    features=path_keypoints,
    matches=path_matches
)

refiner = PixSfM()
reconstruction, sfm_outputs = refiner.triangulation(
    output_dir=path_sfm,
    image_dir=path_images,
    pairs_path=path_pairs,
    features_path=path_keypoints,
    matches_path=path_matches,
    cache_path=path_cache,
)
