from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from cad_engineering.config import ORB_MIN_GOOD_MATCHES, ORB_MIN_INLIER_RATIO


@dataclass
class AlignmentResult:
    image_a: np.ndarray
    aligned_b: np.ndarray
    homography: np.ndarray
    metadata: dict


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _quality_from_mask(mask: np.ndarray | None, match_count: int) -> float:
    if mask is None or match_count == 0:
        return 0.0
    return float(np.count_nonzero(mask) / match_count)


def _compute_homography(points_b: np.ndarray, points_a: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    if len(points_b) < 4 or len(points_a) < 4:
        return None, None
    try:
        # 1. Use OpenCV's C++ RANSAC estimator for translation, rotation, and uniform scale
        matrix, inliers = cv2.estimateAffinePartial2D(points_b, points_a, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if matrix is None:
            return None, None
        
        # 2. Extract scale and translation, and force rotation to exactly 0 degrees
        s = float(np.sqrt(matrix[0, 0]**2 + matrix[1, 0]**2))
        tx = float(matrix[0, 2])
        ty = float(matrix[1, 2])
        
        # 3. Construct the zero-rotation affine matrix and convert to 3x3 homography
        matrix_zero_rot = np.array([[s, 0.0, tx], [0.0, s, ty]], dtype=np.float32)
        homography = np.vstack([matrix_zero_rot, [0, 0, 1]])
        return homography, inliers
    except Exception as exc:
        raise ValueError("OpenCV failed while computing the alignment matrix.") from exc


def _orb_homography(gray_a: np.ndarray, gray_b: np.ndarray) -> tuple[np.ndarray | None, int, float]:
    try:
        detector = cv2.ORB_create(nfeatures=6000)
        kp_a, desc_a = detector.detectAndCompute(gray_a, None)
        kp_b, desc_b = detector.detectAndCompute(gray_b, None)
        if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
            return None, 0, 0.0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.knnMatch(desc_b, desc_a, k=2)
        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        pts_b = np.float32([kp_b[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_a = np.float32([kp_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, mask = _compute_homography(pts_b, pts_a)
        return homography, len(good), _quality_from_mask(mask, len(good))
    except cv2.error as exc:
        raise ValueError(f"OpenCV ORB alignment failed: {exc}") from exc


def _sift_homography(gray_a: np.ndarray, gray_b: np.ndarray) -> tuple[np.ndarray | None, int, float]:
    try:
        detector = cv2.SIFT_create(nfeatures=2000)
        kp_a, desc_a = detector.detectAndCompute(gray_a, None)
        kp_b, desc_b = detector.detectAndCompute(gray_b, None)
        if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
            return None, 0, 0.0
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        matches = matcher.knnMatch(desc_b, desc_a, k=2)
        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.70 * n.distance:
                    good.append(m)
        pts_b = np.float32([kp_b[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_a = np.float32([kp_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, mask = _compute_homography(pts_b, pts_a)
        return homography, len(good), _quality_from_mask(mask, len(good))
    except cv2.error as exc:
        raise ValueError(f"OpenCV SIFT alignment failed: {exc}") from exc


def homography_scale_factor(homography: np.ndarray) -> float:
    if homography is None or homography.shape != (3, 3):
        return 1.0
    h = homography / homography[2, 2] if homography[2, 2] else homography
    scale_x = float(np.sqrt(h[0, 0] ** 2 + h[1, 0] ** 2))
    scale_y = float(np.sqrt(h[0, 1] ** 2 + h[1, 1] ** 2))
    return float((scale_x + scale_y) / 2.0)


def align_drawing_b_to_a(image_a: np.ndarray, image_b: np.ndarray) -> AlignmentResult:
    gray_a = _gray(image_a)
    gray_b = _gray(image_b)

    # 1. Determine downscale factor for feature detection to prevent OutOfMemoryError
    max_dim = 1500
    h_a, w_a = gray_a.shape[:2]
    f = 1.0
    if max(h_a, w_a) > max_dim:
        f = max_dim / max(h_a, w_a)

    if f < 1.0:
        gray_a_small = cv2.resize(gray_a, (0, 0), fx=f, fy=f, interpolation=cv2.INTER_AREA)
        gray_b_small = cv2.resize(gray_b, (0, 0), fx=f, fy=f, interpolation=cv2.INTER_AREA)
    else:
        gray_a_small = gray_a
        gray_b_small = gray_b

    # 2. Run ORB first on downscaled images
    orb_h_small, orb_matches, orb_ratio = _orb_homography(gray_a_small, gray_b_small)
    use_sift = orb_h_small is None or orb_matches < ORB_MIN_GOOD_MATCHES or orb_ratio < ORB_MIN_INLIER_RATIO

    method = "ORB"
    homography_small = orb_h_small
    match_count = orb_matches
    inlier_ratio = orb_ratio
    fallback_reason = None

    if use_sift:
        reasons = []
        if orb_h_small is None:
            reasons.append("homography failed")
        if orb_matches < ORB_MIN_GOOD_MATCHES:
            reasons.append(f"good matches {orb_matches} < {ORB_MIN_GOOD_MATCHES}")
        if orb_ratio < ORB_MIN_INLIER_RATIO:
            reasons.append(f"inlier ratio {orb_ratio:.3f} < {ORB_MIN_INLIER_RATIO:.3f}")
        fallback_reason = "; ".join(reasons)
        
        sift_h_small, sift_matches, sift_ratio = _sift_homography(gray_a_small, gray_b_small)
        if sift_h_small is None:
            raise ValueError(
                "Alignment failed with ORB and SIFT. Try drawings with more shared linework or less cropping."
            )
        method = "SIFT"
        homography_small = sift_h_small
        match_count = sift_matches
        inlier_ratio = sift_ratio

    # 3. Restore translation components to high-res coordinates
    homography = homography_small.copy()
    homography[0, 2] = homography_small[0, 2] / f
    homography[1, 2] = homography_small[1, 2] / f

    # 4. Warp original high-resolution drawing B
    try:
        aligned_b = cv2.warpPerspective(
            image_b,
            homography,
            (image_a.shape[1], image_a.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    except cv2.error as exc:
        raise ValueError("OpenCV failed while warping Drawing B into Drawing A coordinates.") from exc

    scale_factor = homography_scale_factor(homography)
    metadata = {
        "alignment_method": method,
        "good_matches": int(match_count),
        "inlier_ratio": float(inlier_ratio),
        "scale_factor": float(scale_factor),
        "orb_good_matches": int(orb_matches),
        "orb_inlier_ratio": float(orb_ratio),
        "sift_fallback_triggered": bool(use_sift),
        "sift_fallback_reason": fallback_reason,
        "homography": homography.tolist(),
    }
    print(
        f"[Stage 3] Alignment used {method} (downscaled fx={f:.3f}): matches={match_count}, "
        f"inlier_ratio={inlier_ratio:.3f}, scale={scale_factor:.4f}"
    )
    return AlignmentResult(image_a=image_a, aligned_b=aligned_b, homography=homography, metadata=metadata)
