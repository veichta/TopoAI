import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

from src.utils.io import load_mask


def distance_between_lines(S1: np.ndarray, S2: np.ndarray) -> float:
    """Compute distance between two segments.

    Args:
        S1 (np.ndarray): First segment given as [p1, p2] where p1 and p2 are points given in
        homogeneous coordinates.
        S2 (np.ndarray): Second segment given as [p1, p2] where p1 and p2 are points given in
        homogeneous coordinates.

    Returns:
        float: Distance between two segments if they are parallel and overlap, np.inf otherwise.
    """
    L1 = np.cross(S1[0], S1[1])
    L2 = np.cross(S2[0], S2[1])

    # check if lines are parallel
    _, _, w = np.cross(L1, L2)

    if abs(w) > 500 or not check_overlap(S1, S2):
        return np.inf

    C = [L1[1], -L1[0], 1]
    p1 = np.cross(C, L1)
    p2 = np.cross(C, L2)
    p1 = p1 / p1[2]
    p2 = p2 / p2[2]

    return norm(p1[:2] - p2[:2])


def check_overlap(S1, S2):
    """Check if two segments overlap.

    Args:
        S1 (np.ndarray): First segment given as [p1, p2] where p1 and p2 are points given in
        homogeneous coordinates.
        S2 (np.ndarray): Second segment given as [p1, p2] where p1 and p2 are points given in
        homogeneous coordinates.

    Returns:
        bool: True if segments overlap, False otherwise.
    """
    L1 = np.cross(S1[0], S1[1])
    L2 = np.cross(S2[0], S2[1])

    p = get_perpendicular_intersection(L1, L2, S1[0])
    intersects = point_in_segment(S2, p)

    p = get_perpendicular_intersection(L1, L2, S1[1])
    intersects = intersects or point_in_segment(S2, p)

    p = get_perpendicular_intersection(L2, L1, S2[0])
    intersects = intersects or point_in_segment(S1, p)

    p = get_perpendicular_intersection(L2, L1, S2[1])
    return intersects or point_in_segment(S1, p)


def get_perpendicular_intersection(L1: np.ndarray, L2: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Get perpendicular intersection between two lines through a point.

    Args:
        L1 (np.ndarray): First line given in homogeneous coordinates.
        L2 (np.ndarray): Second line given in homogeneous coordinates.
        p (np.ndarray): Point given in homogeneous coordinates.

    Returns:
        np.ndarray: Perpendicular intersection between two lines through a point.
    """
    C = np.array([L1[1], -L1[0], 1])

    C[2] = -C[0] * p[0] - C[1] * p[1]
    p = np.cross(C, L2)
    return p / p[2]


def point_in_segment(S: np.ndarray, p: np.ndarray) -> bool:
    """Check if point is in segment.

    Args:
        S (np.ndarray): Segment given as [p1, p2] where p1 and p2 are points given in
        homogeneous coordinates.
        p (np.ndarray): X coordinate of point.

    Returns:
        bool: True if point is in segment, False otherwise.
    """
    return (norm(p - S[0]) + norm(p - S[1]) - norm(S[0] - S[1])) == 0


def est_road_width(mask: np.ndarray) -> float:
    """Estimate pixels per meter.

    Args:
        mask (np.ndarray): Mask as numpy array.

    Returns:
        float: Estimated pixels per meter.
    """

    # fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # edges = cv2.Canny(mask.copy(), 0, 1, apertureSize=3)
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=40, maxLineGap=3)
    # # plot_lines(mask, edges, lines)
    # ax[0][0].imshow(mask, cmap="gray")
    # ax[1][0].imshow(edges, cmap="gray")
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         ax[1][0].plot([x1, x2], [y1, y2], "r-")

    mask = cv2.GaussianBlur(mask, (13, 13), 0)
    edges = cv2.Canny(mask, 0, 1, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
    # plot_lines(mask, edges, lines)

    # ax[0][1].imshow(mask, cmap="gray")
    # ax[1][1].imshow(edges, cmap="gray")
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         ax[1][1].plot([x1, x2], [y1, y2], "r-")

    # plt.show()
    # exit()

    if lines is not None:
        distances = np.ones((len(lines), len(lines))) * np.inf
        # for each line in lines, find closest parallel line in other lines and compute distance
        # between them
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            l1 = np.cross([x1, y1, 1], [x2, y2, 1])

            for j, line2 in enumerate(lines):
                x3, y3, x4, y4 = line2[0]

                if i == j:
                    continue

                distances[i, j] = distance_between_lines(
                    S1=np.array([[x1, y1, 1], [x2, y2, 1]]),
                    S2=np.array([[x3, y3, 1], [x4, y4, 1]]),
                )

            # for j, line2 in enumerate(lines):
            #     x3, y3, x4, y4 = line2[0]

            #     for idx, line in enumerate(lines):
            #         x1, y1, x2, y2 = line[0]
            #         if idx == i:
            #             plt.plot([x1, x2], [y1, y2], "b-")
            #         elif idx == j:
            #             plt.plot([x1, x2], [y1, y2], "g-")
            #         else:
            #             plt.plot([x1, x2], [y1, y2], "r-")

            #     plt.title(f"Distance: {distances[i, j]}")

            #     plt.show()

            # closest_line = np.argmin(distances[i, :])

            # l2 = lines[closest_line][0]
            # l2 = np.cross([l2[0], l2[1], 1], [l2[2], l2[3], 1])

            # for idx, line in enumerate(lines):
            #     x1, y1, x2, y2 = line[0]
            #     if idx == i:
            #         plt.plot([x1, x2], [y1, y2], "b-")
            #     elif idx == closest_line:
            #         plt.plot([x1, x2], [y1, y2], "g-")
            #     else:
            #         plt.plot([x1, x2], [y1, y2], "r-")

            # plt.title(f"Distance: {distances[i, closest_line]}")
            # plt.show()

        dists = np.argmin(distances, axis=1)
        dists = [d for d in dists if d > 2 and d < 100 and d != np.inf and d != np.nan]
        # print(np.mean(dists))

        # exit()

        return np.mean(dists) if len(dists) > 0 else 0

    return 0


def plot_lines(mask, edges, lines):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(mask, cmap="gray")
    ax[1].imshow(edges, cmap="gray")

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            ax[1].plot([x1, x2], [y1, y2], "r-")
            ax[2].plot([x1, x2], [y1, y2], "r-")

    plt.show()


def est_avg_road_width(mask_paths: list, dataset: str):
    """Estimate average pixels per meter for a dataset.

    Args:
        mask_paths (list): List of paths to masks.

    Returns:
        float: Estimated average pixels per meter.
    """
    masks = [load_mask(path) for path in mask_paths[:1000] if dataset in path]
    # print(next(path for path in mask_paths if dataset in path))
    estimations = [est_road_width(mask) for mask in tqdm(masks)]
    return np.mean([val for val in estimations if val > 0])


def estimate_road_widths(masks: list) -> tuple:
    """Estimate average pixels width of the road masks for each dataset.

    Args:
        masks (list): List of paths to masks.

    Returns:
        tuple: Tuple containing:
            cil_road_width (float): Average road width in pixels for CIL dataset.
            epfl_road_width (float): Average road width in pixels for EPFL dataset.
            roadtracer_road_width (float): Average road width in pixels for RoadTracer dataset.
            deepglobe_road_width (float): Average road width in pixels for DeepGlobe dataset.
            mit_road_width (float): Average road width in pixels for MIT dataset.
    """
    logging.info("Estimating pixels per meter for train dataset.")
    cil_road_width = est_avg_road_width(masks, "cil")

    logging.info("Estimating pixels per meter for EPFL dataset.")
    epfl_road_width = est_avg_road_width(masks, "epfl")

    logging.info("Estimating pixels per meter for RoadTracer dataset.")
    roadtracer_road_width = est_avg_road_width(masks, "roadtracer")

    logging.info("Estimating pixels per meter for DeepGlobe dataset.")
    deepglobe_road_width = est_avg_road_width(masks, "dg")

    logging.info("Estimated average road width (in pixel) for MIT dataset:")
    mit_road_width = est_avg_road_width(masks, "mit")

    logging.info(f"Estimated avg road width (in pixel) for CIL:        {cil_road_width}")
    logging.info(f"Estimated avg road width (in pixel) for EPFL:       {epfl_road_width}")
    logging.info(f"Estimated avg road width (in pixel) for RoadTracer: {roadtracer_road_width}")
    logging.info(f"Estimated avg road width (in pixel) for DeepGlobe:  {deepglobe_road_width}")
    logging.info(f"Estimated avg road width (in pixel) for MIT:        {mit_road_width}")
    return (
        cil_road_width,
        epfl_road_width,
        roadtracer_road_width,
        deepglobe_road_width,
        mit_road_width,
    )
