import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import skimage as ski
import scipy
import skan
import networkx as nx

# memastikan gambar biner (hitam dan putih)
def binarize(image:np.ndarray, limit:int=127) -> np.ndarray:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim > 2 else image
    _, binarized = cv2.threshold(gray_image, limit, 255, cv2.THRESH_BINARY)
    return binarized

# menyimpulkan data statistik
def summarizeData(data:np.ndarray, printTitle:str="") -> dict[str, float]:
    num = len(data)
    mean = np.mean(data)
    sd = np.std(data)
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    minimum = min(data)
    maximum = max(data)

    if len(printTitle) > 0:
        print(f"DATA SUMMARY of [{printTitle}]:")
        print(f"Num     : {num}")
        print(f"Mean    : {mean:.4f}")
        print(f"SD      : {sd:.4f}")
        print(f"SD rel  : {sd/mean*100:.2f}%")
        print(f"Median  : {median:.4f}")
        print(f"MAD     : {mad:.4f}")
        print(f"MAD rel : {mad/median*100:.2f}%")
        print(f"Min     : {minimum:.4f}")
        print(f"Max     : {maximum:.4f}")
        print("")

    return {
        'num': num,
        'mean': mean,
        'sd': sd,
        'median': median, 
        'mad': mad, 
        'min': minimum, 
        'max': maximum
    }

# menentukan jarak antartitik tengah pasangan tetangga
def getNeighborCenterDistance(centroids, label_pair:tuple[int, int]) -> float:
    center1 = centroids[label_pair[0]]
    center2 = centroids[label_pair[1]]
    center_distance = math.dist(center1, center2)
    return center_distance

# menentukan orientasi sudut garis jarak antartitik tengah pasangan tetangga
def getNeighborOrientationDistance(centroids, label_pair:tuple[int, int]) -> np.ndarray:
    center1 = centroids[label_pair[0]]
    center2 = centroids[label_pair[1]]
    orientation = (np.arctan2((center1[1] - center2[1]), center1[0] - center2[0])) % np.pi
    return orientation

def extractContours(binary:np.ndarray) -> list:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return list(contours)

def interpolateCoordinates(coordinates:list, number:int) -> np.ndarray:
    coordinates = np.vstack([coordinates, coordinates[0]])
    index_old = np.linspace(0, 1, len(coordinates))
    index_new = np.linspace(0, 1, number, endpoint=False)

    new_coordinates = np.zeros((number, 2))

    new_coordinates[:, 0] = np.interp(index_new, index_old, coordinates[:, 0])
    new_coordinates[:, 1] = np.interp(index_new, index_old, coordinates[:, 1])

    return new_coordinates

def interpolateContours(contours:list, number:int) -> np.ndarray:
    contours_interpolated = np.zeros((len(contours), number, 2))

    for i in range(len(contours)):
        contours_interpolated[i] = interpolateCoordinates(np.squeeze(contours[i]), 100)

    return contours_interpolated

def standardizeContours(contours_interpolated:np.ndarray) -> np.ndarray:
    contours_standardized = np.zeros_like(contours_interpolated)

    for i in range(len(contours_interpolated)):
        contours_standardized[i] = standardizeContour(contours_interpolated[i])

    return contours_standardized

    # centroids = np.mean(contours_interpolated, axis=1)
    # centered = contours_interpolated - centroids
    # norm = np.linalg.norm(centered, axis=(1, 2))
    # standardized = centered / norm
    # return standardized

def alignContour(reference:np.ndarray, target:np.ndarray) -> tuple[np.ndarray, float]:

    closest_distance = np.inf
    target_rotated = None

    distances_normal = distancesByShifting(reference, target)
    distances_mirrored = distancesByShifting(reference, target[::-1])

    is_mirrored = np.min(distances_normal) > np.min(distances_mirrored)
    best_distances = distances_mirrored if is_mirrored else distances_normal
    best_shift = np.argmin(best_distances)

    target_shifted = np.roll(target[::-1] if is_mirrored else target, -best_shift, axis=0)
    target_rotated = rotateSVD(reference, target_shifted)

    closest_distance = procrustesDistance(reference, target_rotated)

    return target_rotated, closest_distance

def distancesByShifting(reference:np.ndarray, target:np.ndarray) -> np.ndarray:
    shift_distances = np.zeros(len(reference))
    
    for shift in range(len(reference)):
        target_shifted = np.roll(target, -shift, axis=0)
        target_rotated = rotateSVD(reference, target_shifted)
        distance = procrustesDistance(reference, target_rotated)
        shift_distances[shift] = distance
        
    return shift_distances

def procrustesDistance(contour1:np.ndarray, contour2:np.ndarray) -> float:
    return np.sqrt(np.sum((contour1 - contour2) ** 2))

def procrustesDistances(reference:np.ndarray, targets:np.ndarray) -> np.ndarray:
    distances = np.zeros(len(targets))

    for i in range(len(distances)):
        distances[i] = procrustesDistance(reference, targets[i])

    return distances

def rotateSVD(reference:np.ndarray, target:np.ndarray) -> np.ndarray:
    M = np.dot(reference.T, target)
    U, _, Vt = np.linalg.svd(M)
    R = np.dot(Vt.T, U.T)

    target_rotated = np.dot(target, R)

    return target_rotated

def processGPA(
        contours_interpolated:np.ndarray, 
        max_iteration:int=10, 
        tolerance:float=1e-5
    ) -> tuple[np.ndarray, np.ndarray]:

    contours_aligned = standardizeContours(contours_interpolated)

    mean_shape = np.copy(contours_aligned[0])

    for _ in range(max_iteration):

        contour_rotated_list = np.zeros_like(contours_aligned)
        for i in range(len(contours_aligned)):
            contour_rotated, _ = alignContour(mean_shape, contours_aligned[i])
            contour_rotated_list[i] = contour_rotated
        
        new_mean_shape = np.mean(contour_rotated_list, axis=0)
        new_mean_shape = standardizeContour(new_mean_shape)

        mean_shape_distance = procrustesDistance(mean_shape, new_mean_shape)
        
        mean_shape = new_mean_shape
        contours_aligned = contour_rotated_list

        if mean_shape_distance < tolerance:
            break

    print(mean_shape_distance)
    return mean_shape, contours_aligned

def testRotationalSymmetry(contour:np.ndarray) -> np.ndarray:
    reference = np.copy(contour)
    target = np.copy(contour)

    shift_distances = distancesByShifting(reference, target)
    return 1 - shift_distances

def testReflectionalSymmetry(contour:np.ndarray) -> np.ndarray:
    reference = np.copy(contour)
    target = np.copy(contour)

    shift_distances = np.zeros(len(reference))
    
    for shift in range(len(reference)):
        reference_shifted = np.roll(reference, -shift, axis=0)
        target_shifted = np.roll(target, -shift, axis=0)[::-1]
        target_rotated = rotateSVD(reference_shifted, target_shifted)
        distance = procrustesDistance(reference_shifted, target_rotated)
        shift_distances[shift] = distance
    
    return 1 - shift_distances

def testParallel(coordinate_standardized:np.ndarray, max_angle=90):
    angles = np.arange(0, max_angle)

    cos_angles = np.cos(np.deg2rad(angles))
    sin_angles = np.sin(np.deg2rad(angles))

    angle_matrix = np.vstack([cos_angles, -sin_angles])

    test_result = np.dot(coordinate_standardized, angle_matrix)

    angle_grid = np.tile(angles, (len(coordinate_standardized), 1))

    visual_result = np.column_stack((test_result.flatten(), angle_grid.flatten()))

    return visual_result

def rotateToAngle(coordinates_standardized, angle):
    cos_angle = np.cos(np.deg2rad(angle))
    sin_angle = np.sin(np.deg2rad(angle))

    angle_matrix = np.array([
        [sin_angle, cos_angle],
        [cos_angle, -sin_angle]
    ])

    return np.dot(coordinates_standardized, angle_matrix)

def standardizeContour(coordinates:np.ndarray) -> np.ndarray:
    centroid = np.mean(coordinates, axis=0)
    centered = coordinates - centroid
    norm = np.linalg.norm(centered)
    standardized = centered / norm
    return standardized

def createContourMap(shape:tuple[int,int], contours:list) -> np.ndarray:
    contour_map = np.zeros(shape)
    cv2.drawContours(contour_map, contours, -1, (255), 1)
    return contour_map

def contourAreas(contours:list) -> np.ndarray:
    areas = np.zeros(len(contours))

    for i in range(len(contours)):
        areas[i] = cv2.contourArea(contours[i])

    return areas

def contourNorms(contours_interpolated:np.ndarray) -> float:
    return np.linalg.norm(contours_interpolated, axis=(1,2))

def drawStandardizedContour(
        shape:tuple[int,int], 
        contour_standardized:np.ndarray, 
        size_factor:float, 
    ) -> np.ndarray:

    width, height = shape
    contour_centered = (contour_standardized * size_factor)

    centroid_coordinate = (width // 2, height // 2)
    contour = np.int32(np.round(contour_centered + centroid_coordinate))

    contour_cv = np.expand_dims(contour, axis=1)

    canvas = np.zeros(shape)
    cv2.drawContours(canvas, [contour_cv], -1, (255), 1)
    cv2.circle(canvas, centroid_coordinate, 1, (255), -1)

    return canvas

def drawStandardizedCoordinates(
        shape:tuple[int,int], 
        coordinates_standardized:np.ndarray, 
        size_factor:float
    ):

    width, height = shape
    coordinates_centered = (coordinates_standardized * size_factor)

    centroid_coordinate = (width // 2, height // 2)
    coordinates = np.int32(np.round(coordinates_centered + centroid_coordinate))

    canvas = np.zeros(shape)

    canvas[coordinates[:, 0], coordinates[:, 1]] = 255
    canvas = cv2.dilate(canvas, np.ones((3, 3), np.uint8), iterations=3)

    return canvas

def drawRotationalSymmetryAxis(
        shape:tuple[int,int], 
        contour_standardized:np.ndarray, 
        size_factor:float, 
        minimum_symmetricity:float=0.9
    ) -> np.ndarray:

    width, height = shape
    rotational_symmetrism = testRotationalSymmetry(contour_standardized)

    contour_centered = (contour_standardized * size_factor)

    centroid_coordinate = (width // 2, height // 2)
    contour = np.int32(np.round(contour_centered + centroid_coordinate))

    contour_cv = np.expand_dims(contour, axis=1)

    canvas = np.zeros(shape)

    for i in range(len(contour)):
        confidence = int(np.round(rotational_symmetrism[i] * 255))
        confidence = confidence if rotational_symmetrism[i] > minimum_symmetricity else 0
        cv2.line(canvas, centroid_coordinate, contour[i], tuple([confidence]))

    cv2.drawContours(canvas, [contour_cv], -1, (255), 1)
    cv2.circle(canvas, centroid_coordinate, 1, (255), -1)

    return canvas

def drawReflectionalSymmetryAxis(
        shape:tuple[int,int], 
        contour_standardized:np.ndarray, 
        size_factor:float, 
        minimum_symmetricity:float=0.9
    ) -> np.ndarray:

    width, height = shape

    rotational_symmetrism = testReflectionalSymmetry(contour_standardized)

    contour_centered = (contour_standardized * size_factor)

    centroid_coordinate = (width // 2, height // 2)
    contour = np.int32(np.round(contour_centered + centroid_coordinate))

    contour_cv = np.expand_dims(contour, axis=1)

    canvas = np.zeros(shape)

    for i in range(len(contour)):
        confidence = int(np.round(rotational_symmetrism[i] * 255))
        confidence = confidence if rotational_symmetrism[i] > minimum_symmetricity else 0
        cv2.line(canvas, centroid_coordinate, contour[i], tuple([confidence]))

    cv2.drawContours(canvas, [contour_cv], -1, (255), 1)
    cv2.circle(canvas, centroid_coordinate, 1, (255), -1)

    return canvas

# memetakan skeletonisasi elemen-elemen pipih dan bercabang
def skeletonize(binary_image:np.ndarray, min_valid_length:int=2) -> tuple[np.ndarray, skan.Skeleton]:

    skeleton_image = ski.morphology.skeletonize(binary_image, method='lee')

    skeleton_skan = skan.Skeleton(skeleton_image)
    skeleton_skan = pruneSkeleton(skeleton_skan, min_valid_length)
    skeleton_image = imageFromSkan(skeleton_skan)

    return skeleton_image, skeleton_skan

def imageFromSkan(skeleton_skan):
    return np.array((skeleton_skan.path_label_image() > 0) * 255).astype(np.uint8)

def smoothBinary(binary_image:np.ndarray, kernel_size:int=3) -> np.ndarray:
    binary_image = binary_image.astype(float)
    blurred = cv2.GaussianBlur(binary_image, (kernel_size, kernel_size), 0) if kernel_size > 0 else binary_image
    thresholded = binarize(blurred)
    return thresholded

# memetakan ketebalan skeleton
def mapThickness(skeleton_image:np.ndarray, binary_image:np.ndarray, tolerance_size:int=3) -> np.ndarray:
    distance_map = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    if tolerance_size > 0 and tolerance_size % 2 == 1:
        distance_map = cv2.dilate(distance_map, np.ones((tolerance_size, tolerance_size), np.uint8))
    thickness_map = (skeleton_image > 0) * distance_map
    return thickness_map

def findLocalBranchOrientation(branch_coordinates:np.ndarray, window_size:int=5, poly_order:int=3):

    if len(branch_coordinates) <= window_size:
        window_size = len(branch_coordinates) + (len(branch_coordinates) & 2) - 1

    branch_x = branch_coordinates[:, 0]
    branch_y = branch_coordinates[:, 1]

    branch_dx = scipy.signal.savgol_filter(branch_x, window_size, poly_order, deriv=1)
    branch_dy = scipy.signal.savgol_filter(branch_y, window_size, poly_order, deriv=1)

    orientations = np.arctan2(branch_dy, branch_dx)

    orientations[:window_size // 2] = orientations[window_size // 2]
    orientations[-(window_size // 2):] = orientations[-(window_size // 2)]

    return (orientations + np.pi/2) % np.pi

# def findLocalBranchCurvature(branch_coordinates:np.ndarray, window_size:int=5, poly_order:int=3):

#     if len(branch_coordinates) <= window_size:
#         window_size = len(branch_coordinates) + (len(branch_coordinates) & 2) - 1

#     branch_x = branch_coordinates[:, 0]
#     branch_y = branch_coordinates[:, 1]

#     branch_ds = np.sqrt(np.diff(branch_x, prepend=branch_x[0])**2 + np.diff(branch_y, prepend=branch_y[0])**2)
#     branch_s = np.cumsum(branch_ds)

#     branch_dx = scipy.signal.savgol_filter(branch_x, window_size, poly_order, deriv=1)
#     branch_dy = scipy.signal.savgol_filter(branch_y, window_size, poly_order, deriv=1)

#     branch_theta = np.arctan2(branch_dy, branch_dx)
#     branch_theta_unwrapped = np.unwrap(branch_theta)
    
#     branch_dtheta_dt = scipy.signal.savgol_filter(branch_theta_unwrapped, window_size, poly_order, deriv=1)
#     branch_ds_dt = scipy.signal.savgol_filter(branch_s, window_size, poly_order, deriv=1)

#     curvatures = np.abs(branch_dtheta_dt / branch_ds_dt)

#     curvatures[:window_size // 2] = curvatures[window_size // 2]
#     curvatures[-(window_size // 2):] = curvatures[-(window_size // 2)-1]

#     norm_curvatures = curvatures * window_size

#     return norm_curvatures

def findLocalBranchCurvature(branch_coordinates:np.ndarray, window_size:int=5, poly_order:int=3):
    if len(branch_coordinates) <= window_size:
        window_size = len(branch_coordinates) + (len(branch_coordinates) & 2) - 1

    branch_x = branch_coordinates[:, 0]
    branch_y = branch_coordinates[:, 1]

    branch_dx = scipy.signal.savgol_filter(branch_x, window_size, poly_order, deriv=1)
    branch_dy = scipy.signal.savgol_filter(branch_y, window_size, poly_order, deriv=1)

    orientations = np.arctan2(branch_dy, branch_dx)
    orientation_vectors = np.column_stack((np.cos(orientations), np.sin(orientations)))

    windows = np.lib.stride_tricks.sliding_window_view(
        orientation_vectors, window_shape=window_size, axis=0
    ).transpose((0, 2, 1))


    curvatures = np.full(len(orientation_vectors), np.nan)
    curvatures[window_size // 2:-(window_size // 2)] = np.arccos(
        windows[:, 0, 0] * windows[:, -1, 0] + windows[:, 0, 1] * windows[:, -1, 1]
    )

    # print(curvatures)

    # curvatures[:window_size // 2] = curvatures[window_size // 2]
    # curvatures[-(window_size // 2):] = curvatures[-(window_size // 2)-1]

    # curvatures = curvatures / window_size

    return curvatures

# def divideBranches(branch_binary, skeleton_skan, intersection_size:2):
#     skeleton_binary = np.zeros(branch_binary.shape, np.uint16)


def giveZeroCurvature(branch_coordinates:np.ndarray):
    return np.zeros(len(branch_coordinates))

def mapBranchOrientation(skeleton_image:np.ndarray, skeleton_skan:skan.Skeleton, window_size:int=5, poly_order:int=3):
    branch_orientation = np.full(skeleton_image.shape, np.nan, np.float64)

    for path_id in range(skeleton_skan.n_paths):
        path_coordinates = skeleton_skan.path_coordinates(path_id)[1:-1]
        local_branch_orientation = findLocalBranchOrientation(path_coordinates, window_size, poly_order)
        branch_orientation[path_coordinates[:, 0], path_coordinates[:, 1]] = local_branch_orientation

    return branch_orientation

def mapBranchCurvature(skeleton_image:np.ndarray, skeleton_skan:skan.Skeleton, window_size:int=5, poly_order:int=3):
    branch_curvature = np.full(skeleton_image.shape, np.nan, np.float64)

    for path_id in range(skeleton_skan.n_paths):
        
        path_coordinates = skeleton_skan.path_coordinates(path_id)[1:-1]
        local_branch_curvature = (
            giveZeroCurvature(path_coordinates)
            if len(path_coordinates) <= window_size 
            else findLocalBranchCurvature(path_coordinates, window_size, poly_order)
        )
        branch_curvature[path_coordinates[:, 0], path_coordinates[:, 1]] = local_branch_curvature

    return branch_curvature

# memproses algoritma PCA (untuk orientasi dan kelengkungan skeleton)
# def processPCA(coordinates, window_size):
#     if window_size % 2 != 1: raise ValueError("window size must be odd")
#     pad_size = int((window_size - 1) * 0.5)
#     padded_coordinates = extendCoordinates(coordinates, pad_size)
#     windows = ski.util.view_as_windows(padded_coordinates, window_shape=(window_size, 2)).reshape(-1, window_size, 2)
#     means = np.mean(windows, axis=1, keepdims=True)
#     centered = windows - means

#     covariants = np.matmul(centered.transpose(0, 2, 1), centered) / (window_size - 1)

#     eigenvalues, eigenvectors = np.linalg.eigh(covariants)

#     return eigenvalues, eigenvectors

# # memetakan orientasi lokal cabang skeleton
# def mapLocalBranchOrientations(branch_coordinates, window_size=5):
#     _, eigenvectors = processPCA(branch_coordinates, window_size)

#     main_eigenvectors = eigenvectors[:,:,-1]
#     branch_orientations = np.arctan2(main_eigenvectors[:, 0], main_eigenvectors[:, 1])
#     branch_orientations = np.mod(branch_orientations, math.pi)

#     return branch_orientations

# # memetakan kelengkungan lokal cabang skeleton
# def mapLocalBranchCurvatures(branch_coordinates, window_size=11):
#     eigenvalues, _ = processPCA(branch_coordinates, window_size)

#     lambda2 = eigenvalues[:, 0]
#     lambda1 = eigenvalues[:, 1]

#     branch_curvatures = lambda2 / (lambda1 + lambda2)
    
#     normalized_branch_curvatures = 2 * branch_curvatures

#     return normalized_branch_curvatures

# # memetakan kelengkungan global cabang skeleton
# def mapCurvature(skeleton_image, skeleton_skan, prune_branch_ends=0, window_size=5):
#     path_coordinates = np.empty((0, 2), dtype=int)
#     path_curvatures = np.array([])
#     for path_id in range(skeleton_skan.n_paths):
#         current_path_coordinates = np.array(skeleton_skan.path_coordinates(path_id), dtype=int)
#         pruned_path_coordinates = pruneCoordinates(current_path_coordinates, prune_branch_ends)
#         current_path_curvatures = mapLocalBranchCurvatures(pruned_path_coordinates, window_size)
#         path_coordinates = np.concatenate((path_coordinates, pruned_path_coordinates), dtype=int)
#         path_curvatures = np.concatenate((path_curvatures, current_path_curvatures))

#     curvature_map = np.full(skeleton_image.shape, np.nan)
#     curvature_map[path_coordinates[:, 0], path_coordinates[:, 1]] = path_curvatures
    
#     return curvature_map

# # memetakan orientasi global cabang skeleton
# def mapOrientation(skeleton_image, skeleton_skan, prune_branch_ends=0, window_size=5):
#     path_coordinates = np.empty((0, 2), dtype=int)
#     path_orientations = np.array([])
#     for path_id in range(skeleton_skan.n_paths):
#         current_path_coordinates = np.array(skeleton_skan.path_coordinates(path_id), dtype=int)
#         pruned_path_coordinates = pruneCoordinates(current_path_coordinates, prune_branch_ends)
#         current_path_orientations = mapLocalBranchOrientations(pruned_path_coordinates, window_size)
#         path_coordinates = np.concatenate((path_coordinates, pruned_path_coordinates), dtype=int)
#         path_orientations = np.concatenate((path_orientations, current_path_orientations))

#     orientation_map = np.full(skeleton_image.shape, np.nan)
#     orientation_map[path_coordinates[:, 0], path_coordinates[:, 1]] = path_orientations
    
#     return orientation_map

# # membentuk objek SKAN dari skeleton 
# def graphSkeleton(skeleton_image, min_prune=0):
#     skeleton_graph = skan.Skeleton(skeleton_image)

#     if min_prune > 0: skeleton_graph = pruneSkeleton(skeleton_graph, min_prune)

#     return skeleton_graph

# memotong cabang pendek yang muncul karena derau
def pruneSkeleton(skeleton_skan, min_valid_length):
    path_lengths = skeleton_skan.path_lengths()
    valid_prune_ids = np.where(path_lengths < min_valid_length)[0]
    pruned_skeleton_skan = skeleton_skan.prune_paths(valid_prune_ids)
    return pruned_skeleton_skan

# # mengambil data koordinat dalam cabang setelah dipotong ujungnya
# def pruneCoordinates(coordinates, prune_length):
#     if 0 < prune_length < 1:
#         prune_length = int(len(coordinates) * prune_length)
#     if len(coordinates) <= prune_length * 2 or not prune_length:
#         return coordinates
#     pruned_coordinates = coordinates[prune_length : -prune_length]
#     return pruned_coordinates

# # mengambil data koordinat dalam cabang setelah diperpanjang ujungnya
# def extendCoordinates(coordinates, pad_size):
#     if len(coordinates) < 2:
#         return np.pad(coordinates, ((pad_size, pad_size), (0, 0)), mode='edge')
    
#     start_direction = coordinates[0] - coordinates[1]
#     start_steps = np.arange(pad_size, 0, -1).reshape(-1, 1)
#     start_padding = coordinates[0] + start_steps * start_direction

#     end_direction = coordinates[-1] - coordinates[-2]
#     end_steps = np.arange(1, pad_size + 1).reshape(-1, 1)
#     end_padding = coordinates[-1] + end_steps * end_direction

#     return np.vstack([start_padding, coordinates, end_padding])

# menentukan okupansi (luasan) area elemen utama dalam menutupi latar
def getAreaOccupancy(binary_objects:np.ndarray, binary_background:np.ndarray=None) -> float:
    object_area = np.count_nonzero(binary_objects)
    background_area = np.count_nonzero(binary_background) if binary_background is not None else binary_objects.size
    area_occupancy = object_area / background_area
    return area_occupancy

def showLabeledMap(labeled_map:np.ndarray, show_all:bool=False) -> np.ndarray:
    plt.figure(figsize=(10, 10))
    plt.imshow(labeled_map, cmap="jet")
    colorbar = plt.colorbar(shrink=0.5)
    labels = np.unique(labeled_map)
    if show_all:
        colorbar.set_ticks(labels)
    else:
        colorbar.set_ticks([labels[0], labels[-1]])
    return labels