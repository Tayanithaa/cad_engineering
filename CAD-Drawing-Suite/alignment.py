"""
Stage 3: Scale + Alignment Estimation.

Uses SIFT keypoints + FLANN matching + Lowe's ratio test to find
correspondences between v1 and v2, then RANSAC homography to estimate
the transform (including scale) that brings v2 into v1's coordinate frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from config import LOWE_RATIO_TEST, MIN_GOOD_MATCHES, RANSAC_REPROJ_THRESHOLD


@dataclass
class AlignmentResult:
    aligned_v2_gray: np.ndarray
    aligned_v2_color: np.ndarray
    homography: np.ndarray | None
    scale_ratio: float
    good_match_count: int
    low_confidence: bool
    message: str


def _decompose_scale(homography: np.ndarray) -> float:
    """Extract an approximate uniform scale factor from a homography matrix."""
    a, b = homography[0, 0], homography[0, 1]
    c, d = homography[1, 0], homography[1, 1]
    sx = math.hypot(a, c)
    sy = math.hypot(b, d)
    return float((sx + sy) / 2.0)


def estimate_alignment(v1_gray: np.ndarray, v2_gray: np.ndarray, v2_color: np.ndarray) -> AlignmentResult:
    """
    Estimate scale + geometric alignment between v1 (reference) and v2,
    then warp v2 into v1's coordinate frame.
    """
    h1, w1 = v1_gray.shape[:2]

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(v1_gray, None)
    kp2, des2 = sift.detectAndCompute(v2_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return AlignmentResult(
            aligned_v2_gray=cv2.resize(v2_gray, (w1, h1)),
            aligned_v2_color=cv2.resize(v2_color, (w1, h1)),
            homography=None,
            scale_ratio=1.0,
            good_match_count=0,
            low_confidence=True,
            message="Not enough keypoints detected for alignment — falling back to simple resize.",
        )

    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        knn_matches = flann.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)
    except cv2.error:
        knn_matches = []

    good_matches = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < LOWE_RATIO_TEST * n.distance:
            good_matches.append(m)

    good_match_count = len(good_matches)
    low_confidence = good_match_count < MIN_GOOD_MATCHES

    if good_match_count < 4:
        return AlignmentResult(
            aligned_v2_gray=cv2.resize(v2_gray, (w1, h1)),
            aligned_v2_color=cv2.resize(v2_color, (w1, h1)),
            homography=None,
            scale_ratio=1.0,
            good_match_count=good_match_count,
            low_confidence=True,
            message=(
                f"Only {good_match_count} good matches found (need >=4 for homography). "
                "Falling back to simple resize; results may be inaccurate."
            ),
        )

    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)

    if homography is None:
        return AlignmentResult(
            aligned_v2_gray=cv2.resize(v2_gray, (w1, h1)),
            aligned_v2_color=cv2.resize(v2_color, (w1, h1)),
            homography=None,
            scale_ratio=1.0,
            good_match_count=good_match_count,
            low_confidence=True,
            message="RANSAC failed to find a valid homography — falling back to simple resize.",
        )

    scale_ratio = _decompose_scale(homography)

    aligned_v2_gray = cv2.warpPerspective(v2_gray, homography, (w1, h1), flags=cv2.INTER_LINEAR)
    aligned_v2_color = cv2.warpPerspective(v2_color, homography, (w1, h1), flags=cv2.INTER_LINEAR)

    inlier_count = int(mask.sum()) if mask is not None else good_match_count
    message = f"Scale ratio detected: {scale_ratio:.2f}x ({inlier_count}/{good_match_count} inlier matches)."
    if low_confidence:
        message += " Low-confidence alignment: fewer than the recommended minimum good matches were found."

    return AlignmentResult(
        aligned_v2_gray=aligned_v2_gray,
        aligned_v2_color=aligned_v2_color,
        homography=homography,
        scale_ratio=scale_ratio,
        good_match_count=good_match_count,
        low_confidence=low_confidence,
        message=message,
    )
