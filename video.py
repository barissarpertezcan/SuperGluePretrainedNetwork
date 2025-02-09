from pathlib import Path
import cv2
import matplotlib.cm as cm
import torch

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

from utils import object_tracker, compute_homography_filter_outliers
from frame_matching import frame_matcher

def Video(input = "0", save_video = False, source_img = None, resize_source = [200, 200], output_dir = None, image_glob = ['*.png', '*.jpg', '*.jpeg'], skip = 1, max_length = 1000000, resize = [640, 480], weight_type = "indoor", max_keypoints = -1, keypoint_threshold = 0.005, nms_radius = 4, sinkhorn_iterations = 4, match_threshold = 0.7, show_keypoints = True, no_display = False, force_cpu = False):
    if len(resize) == 2 and resize[1] == -1:
        resize = resize[0:1]
    if len(resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            resize[0], resize[1]))
    elif len(resize) == 1 and resize[0] > 0:
        print('Will resize max dimension to {}'.format(resize[0]))
    elif len(resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

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

    vs = VideoStreamer(input, resize, skip,
                       image_glob, max_length)

    if source_img is not None:
        print("\n\n--------target image is given------------\n\n")
        frame = cv2.imread(source_img, 0)
        frame = cv2.resize(frame, (resize_source[0], resize_source[1]), cv2.INTER_AREA)
        h, w = frame.shape
        top_left, top_right, bottom_left, bottom_right = (0, 0), (w-1, 0), (0, h-1), (w-1, h-1) 
    else:
        frame, ret = vs.next_frame()
        assert ret, 'Error when reading the first frame (try different --input?)'

    frame_tensor = frame2tensor(frame, device)
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor
    last_frame = frame
    last_image_id = 0

    if output_dir is not None:
        print('==> Will write outputs to {}'.format(output_dir))
        Path(output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not no_display:
        cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SuperGlue matches', 640*2, 480)
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the anchor\n'
          '\te/r: increase/decrease the keypoint confidence threshold\n'
          '\td/f: increase/decrease the match filtering threshold\n'
          '\tk: toggle the visualization of keypoints\n'
          '\tq: quit')

    timer = AverageTimer()

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('results/output.avi', fourcc, 20.0, (640,  480))

    while True:
        frame, ret = vs.next_frame()
        if not ret:
            print('Finished demo_superglue.py')
            break
        timer.update('data')
        stem0, stem1 = last_image_id, vs.i - 1

        frame_tensor = frame2tensor(frame, device)
        
        # function begins
        pred, kpts0, kpts1, mkpts0, mkpts1, color = frame_matcher(frame_tensor, last_data, matching)
        
        timer.update('forward')

        # if there are enough matches to compute homography, draw the borders of the object that we try to match
        if source_img is not None and len(mkpts0) >= 4:
            M, mkpts0, mkpts1 = compute_homography_filter_outliers(mkpts0, mkpts1)
            frame = object_tracker(frame, M, top_left, top_right, bottom_left, bottom_right)
            
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
            'Image Pair: {:06}:{:06}'.format(stem0, stem1),
        ]

        out = make_matching_plot_fast(
            last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=show_keypoints, small_text=small_text)

        # cv2.imwrite("with_ransac.png", out)
        # break

        if not no_display:
            cv2.imshow('SuperGlue matches', out)
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q':
                vs.cleanup()
                print('Exiting (via q) demo_superglue.py')
                break
            elif key == 'n':  # set the current frame as anchor
                last_data = {k+'0': pred[k+'1'] for k in keys}
                last_data['image0'] = frame_tensor
                last_frame = frame
                last_image_id = (vs.i - 1)
            elif key in ['e', 'r']:
                # Increase/decrease keypoint threshold by 10% each keypress.
                d = 0.1 * (-1 if key == 'e' else 1)
                matching.superpoint.config['keypoint_threshold'] = min(max(
                    0.0001, matching.superpoint.config['keypoint_threshold']*(1+d)), 1)
                print('\nChanged the keypoint threshold to {:.4f}'.format(
                    matching.superpoint.config['keypoint_threshold']))
            elif key in ['d', 'f']:
                # Increase/decrease match threshold by 0.05 each keypress.
                d = 0.05 * (-1 if key == 'd' else 1)
                matching.superglue.config['match_threshold'] = min(max(
                    0.05, matching.superglue.config['match_threshold']+d), .95)
                print('\nChanged the match threshold to {:.2f}'.format(
                    matching.superglue.config['match_threshold']))
            elif key == 'k':
                show_keypoints = not show_keypoints

        if save_video:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # write the output frame
            video_writer.write(frame)

        timer.update('viz')
        timer.print()

        if output_dir is not None:
            #stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
            stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
            out_file = str(Path(output_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, out)
    
    cv2.destroyAllWindows()
    vs.cleanup()