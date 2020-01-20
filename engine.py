#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Lane detection functions
'''

import math
import cv2
import numpy as np
import matplotlib.image as mpimg
from functools import partial
from moviepy.editor import VideoFileClip


def convert_colorspace(img, colorspace='grayscale'):
    """Converts RGB image to another colorspace

    Args:
        img (numpy.ndarray): input RGB image
        colorspace (str, optional): target colorspace

    Returns:
        numpy.ndarray: converted image
    """

    if colorspace == 'grayscale':
        cspace_code = cv2.COLOR_RGB2GRAY
    elif colorspace == 'hsv':
        cspace_code = cv2.COLOR_RGB2HSV
    elif colorspace == 'hsl':
        cspace_code = cv2.COLOR_RGB2HSL

    return cv2.cvtColor(img, cspace_code)


def overlay_mask(img, vertices):
    """Applies masking on image. Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.

    Args:
        img (numpy.ndarray): input image
        vertices (numpy.ndarray): vertices of the mask to apply

    Returns:
        numpy.ndarray: resulting image
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[0, 255, 0], thickness=3):
    """Draw lines on the image inplace

    Args:
        img (numpy.ndarray): input image
        lines (list<tuple>): lines to draw on image
        color (list<int>, optional): color of lines in RGB
        thickness (int, optional): thickness of lines drawn
    """

    for x1, y1, x2, y2 in lines:
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def extrapolate_lines(lines, img_shape, min_slope=0.3):
    """Aggregate detectes lines into two lanes by extrapolation

    Args:
        lines (list<list<tuple>>): detected lines
        img_shape (tuple<int>): image shape
        min_slope (float, optional): minimum slope to consider the edge as being part of the lane

    Returns:
        list<tuple>: list of extrapolated lanes
    """

    extrapolated_lines = []

    # Store slopes
    left_slopes, right_slopes = [], []
    left_intercepts, right_intercepts = [], []
    ymin = img_shape[0]
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y2 - slope * x2
            # Reduce noise for outliers
            if min_slope < abs(slope) < math.inf:
                # Check if left lane
                if slope < 0:
                    # Double check to avoid noise from other part of the image
                    if max(x1, x2) < img_shape[1] / 2:
                        left_slopes.append(slope)
                        left_intercepts.append(intercept)
                else:
                    if min(x1, x2) > img_shape[1] / 2:
                        right_slopes.append(slope)
                        right_intercepts.append(intercept)
                ymin = min(ymin, y1, y2)

    if len(left_slopes) > 0:
        # Average slope and intercept
        left_slope = sum(left_slopes) / len(left_slopes)
        left_intercept = sum(left_intercepts) / len(left_intercepts)
        # Add the extrapolated lane
        left = (int((img_shape[0] - left_intercept) / left_slope), img_shape[0],
                int((ymin - left_intercept) / left_slope), int(ymin))
        extrapolated_lines.append(left)

    if len(right_slopes) > 0:
        right_slope = sum(right_slopes) / len(right_slopes)
        right_intercept = sum(right_intercepts) / len(right_intercepts)
        right = (int((img_shape[0] - right_intercept) / right_slope), img_shape[0],
                 int((ymin - right_intercept) / right_slope), int(ymin))
        extrapolated_lines.append(right)

    return extrapolated_lines


def hough_lines(img, rho=2, theta=np.pi / 180, threshold=20, min_line_len=5, max_line_gap=25, thickness=3):
    """Perform a Hough transform on img

    Args:
        img (numpy.ndarray): input image
        rho (float, optional): distance resolution in pixels of the Hough grid
        theta (float, optional): angular resolution in radians of the Hough grid
        threshold (float, optional): minimum number of votes (intersections in Hough grid cell)
        min_line_len (int, optional): minimum number of pixels making up a line
        max_line_gap (int, optional): maximum gap in pixels between connectable line segments
        thickness (int, optional): thickness of lines drawn on resulting image

    Returns:
        numpy.ndarray: result image
    """
    # Hough transform
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # Line extrapolation
    extrapolated_lines = extrapolate_lines(lines, line_img.shape)
    # Image display
    draw_lines(line_img, extrapolated_lines, thickness=thickness)

    return line_img


def weighted_img(img1, img2, α=0.8, β=1., γ=0.):
    """Interpolate two images into a single one by applying
    α * img1 + β * img2 + γ

    Args:
        img1 (numpy.ndarray[H, W, C]): first image
        img2 (numpy.ndarray[H, W, C]): second image
        α (float, optional): weight of first image
        β (float, optional): weight of second image
        γ (float, optional): offset

    Returns:
        numpy.ndarray: resulting image
    """
    return cv2.addWeighted(img2, α, img1, β, γ)


def get_depth_vertices(img_shape, lat_offset=0.08, horizon=(0.55, 0.5), vert_range=(0.62, 0.9)):
    """Compute depth view vertices

    Args:
        img_shape (tuple<int>): shape of the input image
        lat_offset (float, optional): relative lateral offset of the bottom of the mask
        horizon (tuple<float>, optional): relative coordinates of apparent horizon
        vert_range (tuple<float>, optional): relative range of vertical masking

    Returns:
        numpy.ndarray: vertices of depth view mask
    """

    # Compute cut coordinates
    leftcut_min = lat_offset + (1 - vert_range[0]) / (1 - horizon[0]) * (horizon[1] - lat_offset)
    leftcut_max = lat_offset + (1 - vert_range[1]) / (1 - horizon[0]) * (horizon[1] - lat_offset)

    vertices = np.array([[
        (leftcut_max * img_shape[1], vert_range[1] * img_shape[0]),
        (leftcut_min * img_shape[1], vert_range[0] * img_shape[0]),
        ((1 - leftcut_min) * img_shape[1], vert_range[0] * img_shape[0]),
        ((1 - leftcut_max) * img_shape[1], vert_range[1] * img_shape[0])]], dtype=np.int32)

    return vertices


def _process_image(img, colorspace='hsv', thickness=3, canny_low=50, canny_high=150):
    """Compute the lane mask of an input image and overlay it on input image

    Args:
        img (numpy.ndarray[H, W, C]): input image
        colorspace (str, optional): colorspace to use for canny edge detection
        thickness (int, optional): thickness of lines on result image
        canny_low (int, optional): lower threshold for canny edge detection
        canny_high (int, optional): upper threshold for canny edge detection

    Returns:
        numpy.ndarray[H, W, 3]: lane mask
    """

    # Grayscale conversion
    cspace_img = convert_colorspace(img, colorspace)
    # Gaussian smoothing
    smooth_img = cv2.GaussianBlur(cspace_img, (3, 3), 0)

    # Colorspace masking
    if colorspace == 'hsv':
        yellow_low = np.array([0, 100, 100])
        yellow_high = np.array([50, 255, 255])

        white_low = np.array([20, 0, 180])
        white_high = np.array([255, 80, 255])

        yellow_mask = cv2.inRange(smooth_img, yellow_low, yellow_high)
        white_mask = cv2.inRange(smooth_img, white_low, white_high)

        smooth_img = cv2.bitwise_or(yellow_mask, white_mask)

    # Canny edge detection
    canny_img = cv2.Canny(smooth_img, canny_low, canny_high)

    # Apply depth view masking
    vertices = get_depth_vertices(img.shape)
    masked_edges = overlay_mask(canny_img, vertices)

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lane_img = hough_lines(masked_edges, thickness=thickness)

    # Overlay result on input image
    return weighted_img(img, lane_img)


def process_image(img_path, thickness=3):
    """Read image and detect lanes on it

    Args:
        img_path (str): input image path
        thickness (int, optional): thickness of lines on result image

    Returns:
        numpy.ndarray[H, W, 3]: input image overlayed with result
    """

    img = mpimg.imread(img_path)

    return _process_image(img, thickness=thickness)


def process_video(video_path, output_file, thickness=3):
    """Display lane detection results on input image

    Args:
        video_path (str): input video path
        output_file (str): output video path
        thickness (int, optional): thickness of lines on result image
    """

    video = VideoFileClip(video_path)
    clip = video.fl_image(partial(_process_image, thickness=thickness))
    clip.write_videofile(output_file, audio=False)
