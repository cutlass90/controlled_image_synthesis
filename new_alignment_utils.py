import math
import logging
import numpy as np
import torch
import skimage.transform as tf


def forward_transform(img, src, dst, target_size):
    tform3 = tf.ProjectiveTransform()
    tform3.estimate(src, dst)
    warped = tf.warp(img, tform3, output_shape=(target_size, target_size))
    return np.round(warped*255).astype(np.uint8)


def inverse_transform(img, src, dst, original_image_size):
    tform3 = tf.ProjectiveTransform()
    tform3.estimate(src, dst)
    warped = tf.warp(img, tform3.inverse, output_shape=original_image_size)
    return np.round(warped*255).astype(np.uint8)


def align_image(img, landmarks, target_size, top_margin=0.07):
    aligned_landmarks, align_params = align_landmarks(landmarks, img.shape[:2], target_size, top_margin)
    aligned_landmarks = np.concatenate([aligned_landmarks, np.array([[0, 0], [0, target_size-1], [target_size-1, 0], [target_size-1, target_size-1]], dtype=np.float)], axis=0)
    dealigned_landmarks = dealign_landmarks(align_params, aligned_landmarks, landmarks, [target_size, target_size])
    src = aligned_landmarks
    dst = dealigned_landmarks
    aligned_face = forward_transform(img, src, dst, target_size)
    param = {'src': src.tolist(), 'dst': dst.tolist()}
    return aligned_face, param


def dealign_image(img, src, dst, original_image_size):
    src = np.array(src)
    dst = np.array(dst)
    return inverse_transform(img, src, dst, original_image_size)


def get_src_points(img_size, landmarks, target_size, top_margin=0.07):
    aligned_landmarks, align_params = align_landmarks(landmarks, img_size, target_size, top_margin)
    dealigned_landmarks = dealign_landmarks(align_params, aligned_landmarks, landmarks, [target_size, target_size])
    return dealigned_landmarks


def align_landmarks(landmarks, img_size, target_size, top_margin):
    """ Run image alignment.

    Args:
        landmarks: 2D-ndarray, coordinate of points, each row is a point [x, y]. Note, x corresponds to dim 1 of image
        target_size: int, target size of image, e.g. 256
        top_margin: float, size of image above eyes to include in percents to face size

    Return:
        new_landmarks: 2D-ndarray, new coordinate of points, each row is a point [x, y]. Note, x corresponds to dim 1 of image
    """
    new_landmarks = landmarks.copy()

    new_landmarks = _clip_landmarks(img_size, new_landmarks, verbose=False)
    rotation_angle = _get_rotation_angle(new_landmarks)

    new_landmarks = _move_eye_brows_up(new_landmarks, top_margin)

    resize_rate_x, resize_rate_y = _get_resize_rate(new_landmarks, target_size)

    new_landmarks = _resize_by_rate(new_landmarks, resize_rate_x, resize_rate_y)
    resize_rate = resize_rate_x if resize_rate_x < resize_rate_y else resize_rate_y
    img_size = [s * resize_rate for s in img_size]

    new_landmarks = _rotate_landmarks(img_size, new_landmarks, rotation_angle)
    img_size = _find_rotated_image_size(img_size, rotation_angle)

    new_landmarks = _crop(img_size, new_landmarks, target_size, resize_rate_x, resize_rate_y)

    align_param = {
        'angle': rotation_angle,
        'resize_rate': min(resize_rate_x, resize_rate_y)}

    return new_landmarks, align_param


def dealign_landmarks(align_params, aligned_landmarks, original_landmarks, img_size):
    dealigned_landmarks = aligned_landmarks.copy()
    dealigned_landmarks = _rotate_landmarks(img_size, dealigned_landmarks, -align_params['angle'])
    dealigned_landmarks = dealigned_landmarks / align_params['resize_rate']
    dealigned_landmarks = dealigned_landmarks.copy() + (original_landmarks[8, :] - dealigned_landmarks[8, :])
    return dealigned_landmarks

#######################################################################################################################
def _clip_landmarks(img_size, landmarks, verbose=True):
    if landmarks[:, 0].max() >= img_size[1] or landmarks[:, 1].max() >= img_size[0] or (landmarks < 0).any():
        if verbose:
            logging.info('Warning! Landmarks out of image.')
        landmarks[:, 0] = np.clip(landmarks[:, 0], 0, img_size[1])
        landmarks[:, 1] = np.clip(landmarks[:, 1], 0, img_size[0])
    return landmarks


def _get_rotation_angle(landmarks):
    l_eye, r_eye = get_eyes_coordinate(landmarks)
    eye3 = [(l_eye[0] + r_eye[0]) / 2, (l_eye[1] + r_eye[1]) / 2]
    katet = landmarks[8, 1] - eye3[1]
    if katet == 0:
        katet = 1
    angle = math.degrees(math.atan((eye3[0] - landmarks[8, 0]) / katet))
    return float(angle)


def get_eyes_coordinate(landmarks):
    """ Calculate coordinate of eyes centers.

    Args:
        landmarks: 2D-ndarray, coordinate of points, each row is a point [x, y]. Note, x corresponds to dim 1 of image

    Return:
        tuple:
            l_eye: list of two int, left eye coordinate [x, y]
            r_eye: list of two int, right eye coordinate [x, y]
    """
    l_eye = [0, 0]
    l_eye[0] = sum([point[0] for point in landmarks[36:42]]) / 6
    l_eye[1] = sum([point[1] for point in landmarks[36:42]]) / 6
    r_eye = [0, 0]
    r_eye[0] = sum([point[0] for point in landmarks[42:48]]) / 6
    r_eye[1] = sum([point[1] for point in landmarks[42:48]]) / 6
    return l_eye, r_eye


def _get_resize_rate(mask_landmarks, target_size):
    mask_landmarks[mask_landmarks < 0] = 0
    resize_rate_x = target_size / (mask_landmarks[:, 1].max() - mask_landmarks[:, 1].min())
    resize_rate_y = target_size / (mask_landmarks[:, 0].max() - mask_landmarks[:, 0].min())
    return resize_rate_x, resize_rate_y


def _resize_by_rate(landmarks, resize_rate_x, resize_rate_y):
    resize_rate = resize_rate_x if resize_rate_x < resize_rate_y else resize_rate_y
    landmarks[landmarks < 0] = 0
    return landmarks * resize_rate


def _find_rotated_image_size(image_size, angle):
    a = np.deg2rad(angle)
    w, h = image_size
    return int(abs(w * np.cos(a)) + abs(h * np.sin(a))), int(
        abs(w * np.sin(a)) + abs(h * np.cos(a)))


def _rotate_landmarks(image_size, landmarks, angle):
    org_center = (np.array(image_size[::-1]) - 1) / 2.
    im_rot_size = _find_rotated_image_size(image_size, angle)
    rot_center = (np.array(im_rot_size[::-1]) - 1) / 2.

    org_x = landmarks[:, 0] - org_center[0]
    org_y = landmarks[:, 1] - org_center[1]
    a = np.deg2rad(angle)
    new_landmarks = np.stack([org_x * np.cos(a) + org_y * np.sin(a),
                              -org_x * np.sin(a) + org_y * np.cos(a)], axis=1)
    new_landmarks[:, 0] += rot_center[0]
    new_landmarks[:, 1] += rot_center[1]
    return new_landmarks


def _move_eye_brows_up(landmarks, top_margin):
    mask_landmarks = landmarks.copy()
    mask_landmarks[17:27, 1] -= (top_margin * (mask_landmarks[8, 1] - mask_landmarks[17:27, 1]))
    return mask_landmarks


def _crop(img_size, mask_landmarks, target_size, resize_rate_x, resize_rate_y):
    c = mask_landmarks[8, 0]
    x0 = c - target_size // 2
    x1 = c + target_size // 2
    x0 = max(x0, 0)
    x1 = max(x1, 0)
    x0 = min(x0, img_size[1])
    x1 = min(x1, img_size[1])
    y0 = max(mask_landmarks[:, 1].min(), 0)
    y1 = y0 + target_size
    if resize_rate_y < resize_rate_x:
        y1 = min(y1, img_size[0])
    cropped_landmarks = mask_landmarks - [x0, y0]
    cropped_landmarks[cropped_landmarks < 0] = 0
    cropped_landmarks[cropped_landmarks[:, 0] >= (x1 - x0), 0] = (x1 - x0 - 1)
    cropped_landmarks[cropped_landmarks[:, 1] >= (y1 - y0), 1] = (y1 - y0 - 1)
    return cropped_landmarks


def hwc2chw(img):
    if len(img.shape) == 3:
        img = np.transpose(img, [2, 0, 1])
    elif len(img.shape) == 4:
        img = np.transpose(img, [0, 3, 1, 2])
    else:
        raise ValueError
    return img


def img2tensor(img):
    img = hwc2chw(img)
    img = (img - 127.5) / 128
    return torch.from_numpy(img).float()


def tensor2img(tensor):
    img = tensor.detach().cpu().numpy()
    img = (img + 1) * 127.5
    img = np.transpose(img, [1, 2, 0])
    return img.astype(np.uint8)


