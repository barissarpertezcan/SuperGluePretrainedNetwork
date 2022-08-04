from pathlib import Path
import cv2
import matplotlib.cm as cm
import torch

from models.matching import Matching
from models.utils import (make_matching_plot_fast, frame2tensor)

import os
from utils import object_tracker, compute_homography_filter_outliers


def frame_matcher_stand_alone(source_img, target_img, resize_source = [640, 480], resize_target = [640, 480], output_dir = None, weight_type = "indoor", max_keypoints = -1, keypoint_threshold = 0.005, nms_radius = 4, sinkhorn_iterations = 4, match_threshold = 0.7, show_keypoints = True, no_display = False, force_cpu = False):

    device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
        'superglue': {
            'weights': weight_type,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    frame = cv2.imread(source_img, 0)
    frame = cv2.resize(frame, (resize_source[0], resize_source[1]), cv2.INTER_AREA)

    target_img = cv2.imread(target_img, 0)
    target_img = cv2.resize(target_img, (resize_target[0], resize_target[1]), cv2.INTER_AREA)

    frame_tensor = frame2tensor(frame, device)
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor
    last_frame = frame

    if output_dir is not None:
        print('==> Will write outputs to {}'.format(output_dir))
        Path(output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not no_display:
        cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SuperGlue matches', 640*2, 480)
    else:
        print('Skipping visualization, will not show a GUI.')

    # added
    h, w = frame.shape
    # top_left, top_right, bottom_left, bottom_right = (0, 0), (w-1, 0), (0, h-1), (w-1, h-1)
    top_left, top_right, bottom_left, bottom_right = (104, 21), (560, 25), (132, 416), (545, 432) # erikli source
    # top_left, top_right, bottom_left, bottom_right = (142, 56), (525, 75), (52, 420), (543, 439) # book source

    frame = target_img

    frame_tensor = frame2tensor(frame, device)
    pred = matching({**last_data, 'image1': frame_tensor})
    kpts0 = last_data['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    color = cm.jet(confidence[valid])

    # if there are enough matches to compute homography
    if len(mkpts0) >= 4:
        M, mkpts0, mkpts1 = compute_homography_filter_outliers(mkpts0, mkpts1)
        object_tracker(frame, M, top_left, top_right, bottom_left, bottom_right)

    # print(len(mkpts0), len(mkpts1))

    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0))
    ]
    k_thresh = matching.superpoint.config['keypoint_threshold']
    m_thresh = matching.superglue.config['match_threshold']
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
    ]
    out = make_matching_plot_fast(
        last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
        path=None, show_keypoints=show_keypoints, small_text=small_text)

    if not no_display:
        cv2.imshow('SuperGlue matches', out)
        key = chr(cv2.waitKey(0) & 0xFF)
        if key == 'q':
            print('Exiting (via q) demo_superglue.py')
            cv2.destroyAllWindows()

    if output_dir is not None:
        cv2.imwrite(os.path.join(output_dir, "matched_features.png"), out)

    return mkpts0, mkpts1

def frame_matcher(frame_tensor, last_data, matching):
    pred = matching({**last_data, 'image1': frame_tensor})
    kpts0 = last_data['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    color = cm.jet(confidence[valid])

    return pred, kpts0, kpts1, mkpts0, mkpts1, color