from numba import njit
import numpy as np
from numpy import ndarray


@njit
def cross_product(v1: ndarray, v2: ndarray):
    return v1[0] * v2[1] - v1[1] * v2[0]


@njit
def line_intersect_distance(p: ndarray, p2: ndarray, q: ndarray, q2: ndarray):
    r = p2 - p
    s = q2 - q

    q_p_diff = q - p
    u_numerator = cross_product(q_p_diff, r)
    denominator = cross_product(r, s)

    # Lines are parallel
    if denominator == 0:
        return 0.

    u = u_numerator / denominator
    t = cross_product(q_p_diff, s) / denominator

    # t is the fraction of the distance along p_vec = p2-p1
    if 0 <= t <= 1 and 0 <= u <= 1:
        return t
    return 0.


@njit
def dist(arr1, arr2):
    return np.sqrt(np.sum((arr1 - arr2) ** 2))


@njit
def line_point_collision(xy1: ndarray, xy2: ndarray, p: ndarray, buffer: float = 0.25):
    d1 = dist(p, xy1)
    d2 = dist(p, xy2)
    line_len = dist(xy1, xy2)

    return True if line_len-buffer <= d1+d2 <= line_len+buffer else False


@njit
def get_node_economic(nodes: ndarray, position: ndarray):
    n_node_groups = int(np.sqrt(nodes.shape[0]))
    half_n_groups = n_node_groups // 2
    is_odd_nr_groups = n_node_groups % 2
    main_node_indices = np.arange(n_node_groups) * n_node_groups + half_n_groups
    cluster_index = np.argmin(np.sum((nodes[main_node_indices] - position) ** 2, axis=1))

    start_index = cluster_index - half_n_groups
    end_index = cluster_index + half_n_groups + is_odd_nr_groups
    min_cluster_dist = np.inf
    min_cluster_index, next_min_cluster_index = 0, 0
    distances = np.sqrt(np.sum((nodes[start_index: end_index] - position) ** 2, axis=1))

    for d, index in zip(distances, np.arange(start_index, end_index)):
        if d < min_cluster_dist:
            next_min_cluster_index = min_cluster_index
            min_cluster_dist = d
            min_cluster_index = index

    return min((min_cluster_index, next_min_cluster_index))


@njit
def get_node(nodes: ndarray, position: ndarray):
    indices = np.argsort(np.sum((nodes - position) ** 2, axis=1))[:2]
    return min(indices)


@njit
def rotate(angles: ndarray, vector: ndarray):
    new_vectors = np.zeros((angles.size, 2))
    rotation_matrix = np.zeros((2, 2))
    cos = np.cos(angles)
    sin = np.sin(angles)
    for i in np.arange(angles.size):
        rotation_matrix[0, 0] = rotation_matrix[1, 1] = cos[i]
        rotation_matrix[0, 1] = -sin[i]
        rotation_matrix[1, 0] = sin[i]
        new_vectors[i] = np.dot(rotation_matrix, vector).flatten()

    return new_vectors


@njit
def car_lines(pos, angle, width, length):
    lines = np.zeros((8, 2))
    cos, sin = np.cos(angle), np.sin(angle)
    width_vector = np.array([-sin, cos]) * width
    length_vector = np.array([cos, sin]) * length

    left_back_corner = pos + width_vector
    right_back_corner = pos - width_vector
    left_front_corner = left_back_corner + length_vector
    right_front_corner = right_back_corner + length_vector

    lines[0], lines[1] = left_back_corner, right_back_corner
    lines[2], lines[3] = left_back_corner, left_front_corner
    lines[4], lines[5] = right_back_corner, right_front_corner
    lines[6], lines[7] = left_front_corner, right_front_corner

    return lines


def get_lidar_lines(racer):
    node_index = get_node(racer.track_nodes[:-1], racer.position)
    dist_to_node = np.sqrt(np.sum((racer.position - racer.track_nodes[node_index]) ** 2))
    vector = np.array([np.cos(racer.angle), np.sin(racer.angle)])
    vector /= np.sqrt(np.sum(vector ** 2))
    car_front = racer.position + vector * racer.car_length
    vector *= racer.diag
    new_vectors = rotate(racer.angles, vector.reshape((2, 1)))
    new_vectors = np.append(new_vectors, vector.reshape((1, 2)), axis=0)
    xs = np.zeros(new_vectors.shape[0] * 2)
    ys = np.zeros(new_vectors.shape[0] * 2)
    angle = np.arctan2(vector[1], vector[0])
    car_bounds = car_lines(racer.position, racer.angle, racer.car_width, racer.car_length)

    for i, vec in enumerate(new_vectors):
        this_i = i * 2
        xs[this_i], ys[this_i] = car_front
        xs[this_i + 1], ys[this_i + 1] = car_front + vec

        for node_id in np.append(np.arange(node_index - 1, racer.track_nodes.shape[0] - 1), np.arange(0, node_index - 1)):
            distance_frac = line_intersect_distance(car_front, car_front + vec,
                                                    racer.track_outer[node_id], racer.track_outer[node_id + 1])
            if distance_frac != 0:
                this_i = i * 2
                xs[this_i + 1], ys[this_i + 1] = car_front + vec * distance_frac
                break
            else:
                distance_frac = line_intersect_distance(car_front, car_front + vec,
                                                        racer.track_inner[node_id], racer.track_inner[node_id + 1])
                if distance_frac != 0:
                    this_i = i * 2
                    xs[this_i + 1], ys[this_i + 1] = car_front + vec * distance_frac
                    break

    car_collides = True if dist_to_node > racer.lane_width else False
    if not car_collides:
        for i in np.arange(3):
            if line_intersect_distance(car_bounds[2 * i], car_bounds[2 * i + 1], racer.track_outer[node_index],
                                       racer.track_outer[node_index + 1]):
                car_collides = True
                break
            elif line_intersect_distance(car_bounds[2 * i], car_bounds[2 * i + 1], racer.track_inner[node_index],
                                         racer.track_inner[node_index + 1]):
                car_collides = True
                break

    return xs, ys, car_bounds, car_collides
