import math

import numpy as np
from skimage.morphology import skeletonize

#!/usr/bin/env python3


"""
The methods are taken from
https://github.com/yxdragon/sknw
"""

import networkx as nx
import numpy as np


# get neighbors d index
def neighbors(shape):
    dim = len(shape)
    block = np.ones([3] * dim)
    block[tuple([1] * dim)] = 0
    idx = np.where(block > 0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx - [1] * dim)
    acc = np.cumprod((1,) + shape[::-1][:-1])
    return np.dot(idx, acc[::-1])


# @jit(nopython=True)  # my mark
def mark(img, nbs):  # mark the array use (0, 1, 2)
    img = img.ravel()
    for p in range(len(img)):
        if img[p] == 0:
            continue
        s = sum(img[p + dp] != 0 for dp in nbs)
        img[p] = 1 if s == 2 else 2


# @jit(nopython=True)  # trans index to r, c...
def idx2rc(idx, acc):
    rst = np.zeros((len(idx), len(acc)), dtype=np.int16)
    for i in range(len(idx)):
        for j in range(len(acc)):
            rst[i, j] = idx[i] // acc[j]
            idx[i] -= rst[i, j] * acc[j]
    rst -= 1
    return rst


# @jit(nopython=True)  # fill a node (may be two or more points)
def fill(img, p, num, nbs, acc, buf):
    back = img[p]
    img[p] = num
    buf[0] = p
    cur = 0
    s = 1

    while True:
        p = buf[cur]
        for dp in nbs:
            cp = p + dp
            if img[cp] == back:
                img[cp] = num
                buf[s] = cp
                s += 1
        cur += 1
        if cur == s:
            break
    return idx2rc(buf[:s], acc)


# trace the edge and use a buffer, then buf.copy, if use [] numba not works
# @jit(nopython=True)
def trace(img, p, nbs, acc, buf):
    c1 = 0
    c2 = 0
    newp = 0
    cur = 1
    while True:
        buf[cur] = p
        img[p] = 0
        cur += 1
        for dp in nbs:
            cp = p + dp
            if img[cp] >= 10:
                if c1 == 0:
                    c1 = img[cp]
                    buf[0] = cp
                else:
                    c2 = img[cp]
                    buf[cur] = cp
            if img[cp] == 1:
                newp = cp
        p = newp
        if c2 != 0:
            break
    return (c1 - 10, c2 - 10, idx2rc(buf[: cur + 1], acc))


# @jit(nopython=True)  # parse the image then get the nodes and edges
def parse_struc(img, pts, nbs, acc):
    img = img.ravel()
    buf = np.zeros(131072, dtype=np.int64)
    num = 10
    nodes = []
    for p in pts:
        if img[p] == 2:
            nds = fill(img, p, num, nbs, acc, buf)
            num += 1
            nodes.append(nds)
    edges = []
    for p in pts:
        for dp in nbs:
            if img[p + dp] == 1:
                edge = trace(img, p + dp, nbs, acc, buf)
                edges.append(edge)
    return nodes, edges


# use nodes and edges build a networkx graph


def build_graph(nodes, edges, multi=False):
    graph = nx.MultiGraph() if multi else nx.Graph()
    for i in range(len(nodes)):
        graph.add_node(i, pts=nodes[i], o=nodes[i].mean(axis=0))
    for s, e, pts in edges:
        ln = np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum()
        graph.add_edge(s, e, pts=pts, weight=ln)
    return graph


def buffer(ske):
    buf = np.zeros(tuple(np.array(ske.shape) + 2), dtype=np.uint16)
    buf[tuple([slice(1, -1)] * buf.ndim)] = ske
    return buf


def mark_node(ske):
    buf = buffer(ske)
    nbs = neighbors(buf.shape)
    acc = np.cumprod((1,) + buf.shape[::-1][:-1])[::-1]
    mark(buf, nbs)
    return buf


def build_sknw(ske, multi=False):
    buf = buffer(ske)
    nbs = neighbors(buf.shape)
    acc = np.cumprod((1,) + buf.shape[::-1][:-1])[::-1]
    mark(buf, nbs)
    pts = np.array(np.where(buf.ravel() == 2))[0]
    nodes, edges = parse_struc(buf, pts, nbs, acc)
    return build_graph(nodes, edges, multi)


# draw the graph


def draw_graph(img, graph, cn=255, ce=128):
    acc = np.cumprod((1,) + img.shape[::-1][:-1])[::-1]
    img = img.ravel()
    for (s, e) in graph.edges():
        eds = graph[s][e]
        if isinstance(graph, nx.MultiGraph):
            for i in eds:
                pts = eds[i]["pts"]
                img[np.dot(pts, acc)] = ce
        else:
            img[np.dot(eds["pts"], acc)] = ce
    for idx in graph.nodes():
        pts = graph.nodes[idx]["pts"]
        img[np.dot(pts, acc)] = cn


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
        ]
    )

    node_img = mark_node(img)
    graph = build_sknw(img)

    plt.imshow(node_img[1:-1, 1:-1], cmap="gray")

    # draw edges by pts
    for (s, e) in graph.edges():
        ps = graph[s][e]["pts"]
        plt.plot(ps[:, 1], ps[:, 0], "green")

    # draw node by o
    nodes = graph.nodes()
    ps = np.array([nodes[i]["o"] for i in nodes])
    plt.plot(ps[:, 1], ps[:, 0], "r.")

    # title and show
    plt.title("Build Graph")
    plt.show()
"""
The Ramer-Douglas-Peucker algorithm roughly ported from the pseudo-code provided
by http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm

The code is taken from
https://github.com/mitroadmaps/roadtracer/blob/master/lib/discoverlib/rdp.py
"""

from math import sqrt


def distance(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def point_line_distance(point, start, end):
    """
    Calaculate the prependicuar distance of given point from the line having
    start and end points.
    """
    if start == end:
        return distance(point, start)

    n = abs(
        (end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1])
    )
    d = sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    return n / d


def rdp(points, epsilon):
    """
    Reduces a series of points to a simplified version that loses detail, but
    maintains the general shape of the series.

    @param points: Series of points for a line geometry represnted in graph.
    @param epsilon: Tolerance required for RDP algorithm to aproximate the
                    line geometry.

    @return: Aproximate series of points for approximate line geometry
    """
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    return (
        rdp(points[: index + 1], epsilon)[:-1] + rdp(points[index:], epsilon)
        if dmax >= epsilon
        else [points[0], points[-1]]
    )


def simplify_edge(ps, max_distance=1):
    """
    Combine multiple points of graph edges to line segments
    so distance from points to segments <= max_distance

    @param ps: array of points in the edge, including node coordinates
    @param max_distance: maximum distance, if exceeded new segment started

    @return: ndarray of new nodes coordinates
    """
    res_points = []
    cur_idx = 0
    for i in range(1, len(ps) - 1):
        segment = ps[cur_idx : i + 1, :] - ps[cur_idx, :]
        angle = -math.atan2(segment[-1, 1], segment[-1, 0])
        ca = math.cos(angle)
        sa = math.sin(angle)
        # rotate all the points so line is alongside first column coordinate
        # and the second col coordinate means the distance to the line
        segment_rotated = np.array([[ca, -sa], [sa, ca]]).dot(segment.T)
        distance = np.max(np.abs(segment_rotated[1, :]))
        if distance > max_distance:
            res_points.append(ps[cur_idx, :])
            cur_idx = i
    if not res_points:
        res_points.append(ps[0, :])
    res_points.append(ps[-1, :])

    return np.array(res_points)


def simplify_graph(graph, max_distance=1):
    """
    @params graph: MultiGraph object of networkx
    @return: simplified graph after applying RDP algorithm.
    """
    all_segments = []
    # Iterate over Graph Edges
    for (s, e) in graph.edges():
        for _, val in graph[s][e].items():
            # get all pixel points i.e. (x,y) between the edge
            ps = val["pts"]
            # create a full segment
            full_segments = np.row_stack([graph.nodes[s]["o"], ps, graph.nodes[e]["o"]])
            # simply the graph.
            segments = rdp(full_segments.tolist(), max_distance)
            all_segments.append(segments)

    return all_segments


def segment_to_linestring(segment):
    """
    Convert Graph segment to LineString require to calculate the APLS mteric
    using utility tool provided by Spacenet.
    """

    if len(segment) < 2:
        return []
    linestring = "LINESTRING ({})"
    sublinestring = ""
    for i, node in enumerate(segment):
        if i == 0:
            sublinestring = sublinestring + "{:.1f} {:.1f}".format(node[1], node[0])
        else:
            if node[0] == segment[i - 1][0] and node[1] == segment[i - 1][1]:
                if len(segment) == 2:
                    return []
                continue
            if i > 1 and node[0] == segment[i - 2][0] and node[1] == segment[i - 2][1]:
                continue
            sublinestring = sublinestring + ", {:.1f} {:.1f}".format(node[1], node[0])
    return linestring.format(sublinestring)


def segmets_to_linestrings(segments):
    """
    Convert multiple segments to LineStrings require to calculate the APLS mteric
    using utility tool provided by Spacenet.
    """

    linestrings = []
    for segment in segments:
        linestring = segment_to_linestring(segment)
        if len(linestring) > 0:
            linestrings.append(linestring)
    if not linestrings:
        linestrings = ["LINESTRING EMPTY"]
    return linestrings


def unique(list1):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def getKeypoints(mask, thresh=0.8, is_gaussian=True, is_skeleton=False, smooth_dist=4):
    """
    Generate keypoints for binary prediction mask.

    @param mask: Binary road probability mask
    @param thresh: Probability threshold used to cnvert the mask to binary 0/1 mask
    @param gaussian: Flag to check if the given mask is gaussian/probability mask
                    from prediction
    @param is_skeleton: Flag to perform opencv skeletonization on the binarized
                        road mask
    @param smooth_dist: Tolerance parameter used to smooth the graph using
                        RDP algorithm

    @return: return ndarray of road keypoints
    """

    if is_gaussian:
        mask /= 255.0
        mask[mask < thresh] = 0
        mask[mask >= thresh] = 1

    h, w = mask.shape
    ske = mask if is_skeleton else skeletonize(mask).astype(np.uint16)
    graph = build_sknw(ske, multi=True)

    segments = simplify_graph(graph, smooth_dist)
    linestrings_1 = segmets_to_linestrings(segments)
    linestrings = unique(linestrings_1)

    keypoints = []
    for line in linestrings:
        linestring = line.rstrip("\n").split("LINESTRING ")[-1]
        points_str = linestring.lstrip("(").rstrip(")").split(", ")
        ## If there is no road present
        if "EMPTY" in points_str:
            return keypoints
        points = []
        for pt_st in points_str:
            x, y = pt_st.split(" ")
            x, y = float(x), float(y)
            points.append([x, y])

            x1, y1 = points[0]
            x2, y2 = points[-1]
            zero_dist1 = math.sqrt((x1) ** 2 + (y1) ** 2)
            zero_dist2 = math.sqrt((x2) ** 2 + (y2) ** 2)

            if zero_dist2 > zero_dist1:
                keypoints.append(points[::-1])
            else:
                keypoints.append(points)
    return keypoints


def getVectorMapsAngles(shape, keypoints, theta=5, bin_size=10):
    """
    Convert Road keypoints obtained from road mask to orientation angle mask.
    Reference: Section 3.1
        https://anilbatra2185.github.io/papers/RoadConnectivityCVPR2019.pdf

    @param shape: Road Label/PIL image shape i.e. H x W
    @param keypoints: road keypoints generated from Road mask using
                        function getKeypoints()
    @param theta: thickness width for orientation vectors, it is similar to
                    thicknes of road width with which mask is generated.
    @param bin_size: Bin size to quantize the Orientation angles.

    @return: Retun ndarray of shape H x W, containing orientation angles per pixel.
    """

    im_h, im_w = shape
    vecmap = np.zeros((im_h, im_w, 2), dtype=np.float32)
    vecmap_angles = np.zeros((im_h, im_w), dtype=np.float32)
    vecmap_angles.fill(360)
    height, width, channel = vecmap.shape
    for j in range(len(keypoints)):
        for i in range(1, len(keypoints[j])):
            a = keypoints[j][i - 1]
            b = keypoints[j][i]
            ax, ay = a[0], a[1]
            bx, by = b[0], b[1]
            bax = bx - ax
            bay = by - ay
            norm = math.sqrt(1.0 * bax * bax + bay * bay) + 1e-9
            bax /= norm
            bay /= norm

            min_w = max(int(round(min(ax, bx) - theta)), 0)
            max_w = min(int(round(max(ax, bx) + theta)), width)
            min_h = max(int(round(min(ay, by) - theta)), 0)
            max_h = min(int(round(max(ay, by) + theta)), height)

            for h in range(min_h, max_h):
                for w in range(min_w, max_w):
                    px = w - ax
                    py = h - ay
                    dis = abs(bax * py - bay * px)
                    if dis <= theta:
                        vecmap[h, w, 0] = bax
                        vecmap[h, w, 1] = bay
                        _theta = math.degrees(math.atan2(bay, bax))
                        vecmap_angles[h, w] = (_theta + 360) % 360

    vecmap_angles = (vecmap_angles / bin_size).astype(int)
    return vecmap, vecmap_angles


def convertAngles2VecMap(shape, vecmapAngles):
    """
    Helper method to convert Orientation angles mask to Orientation vectors.

    @params shape: Road mask shape i.e. H x W
    @params vecmapAngles: Orientation agles mask of shape H x W
    @param bin_size: Bin size to quantize the Orientation angles.

    @return: ndarray of shape H x W x 2, containing x and y values of vector
    """

    h, w = shape
    vecmap = np.zeros((h, w, 2), dtype=np.float)

    for h1 in range(h):
        for w1 in range(w):
            angle = vecmapAngles[h1, w1]
            if angle < 36.0:
                angle *= 10.0
                if angle >= 180.0:
                    angle -= 360.0
                vecmap[h1, w1, 0] = math.cos(math.radians(angle))
                vecmap[h1, w1, 1] = math.sin(math.radians(angle))

    return vecmap


def convertVecMap2Angles(shape, vecmap, bin_size=10):
    """
    Helper method to convert Orientation vectors to Orientation angles.

    @params shape: Road mask shape i.e. H x W
    @params vecmap: Orientation vectors of shape H x W x 2

    @return: ndarray of shape H x W, containing orientation angles per pixel.
    """

    im_h, im_w = shape
    angles = np.zeros((im_h, im_w), dtype=np.float)
    angles.fill(360)

    for h in range(im_h):
        for w in range(im_w):
            x = vecmap[h, w, 0]
            y = vecmap[h, w, 1]
            angles[h, w] = (math.degrees(math.atan2(y, x)) + 360) % 360

    angles = (angles / bin_size).astype(int)
    return angles
