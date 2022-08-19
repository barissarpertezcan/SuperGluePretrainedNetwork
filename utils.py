import cv2
import numpy as np

def object_tracker(frame, M, top_left, top_right, bottom_left, bottom_right):

    target_top_left = np.matmul(M, (*top_left, 1))
    target_top_right = np.matmul(M, (*top_right, 1))
    target_bottom_left = np.matmul(M, (*bottom_left, 1))
    target_bottom_right = np.matmul(M, (*bottom_right, 1))
    
    # print(target_top_right)

    target_top_left = target_top_left[:2] / target_top_left[2]
    target_top_right = target_top_right[:2] / target_top_right[2] 
    target_bottom_left = target_bottom_left[:2] / target_bottom_left[2]
    target_bottom_right = target_bottom_right[:2] / target_bottom_right[2] 
    
    target_top_left = target_top_left.astype(int)
    target_top_right = target_top_right.astype(int)
    target_bottom_left = target_bottom_left.astype(int)
    target_bottom_right = target_bottom_right.astype(int)

    circle_color = (0, 255, 0)
    frame = cv2.circle(frame, target_top_left, 10, circle_color, -1)
    frame = cv2.circle(frame, target_top_right, 10, circle_color, -1)
    frame = cv2.circle(frame, target_bottom_left, 10, circle_color, -1)
    frame = cv2.circle(frame, target_bottom_right, 10, circle_color, -1)

    line_color = (0, 0, 255)
    # frame = cv2.rectangle(frame, (target_bottom_left[0], target_bottom_left[1]), (target_top_right[0], target_top_right[1]), (0, 255, 0), 2)
    frame = cv2.line(frame, target_top_left, target_top_right, line_color, 2)
    frame = cv2.line(frame, target_top_left, target_bottom_left, line_color, 2)
    frame = cv2.line(frame, target_bottom_left, target_bottom_right, line_color, 2)
    frame = cv2.line(frame, target_bottom_right, target_top_right, line_color, 2)

    center = (int((target_top_left[0] + target_bottom_right[0]) / 2), int((target_top_left[1] + target_bottom_right[1]) / 2))
    frame = cv2.putText(frame, "object", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return frame

def compute_homography_filter_outliers(mkpts0, mkpts1):
    M, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
    matchesMask = mask.astype(bool).ravel().tolist()

    mkpts0 = mkpts0[matchesMask]
    mkpts1 = mkpts1[matchesMask]
    return M, mkpts0, mkpts1