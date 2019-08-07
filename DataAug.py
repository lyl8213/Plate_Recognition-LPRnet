import random
import numpy as np
import cv2


def jitter(img, jitter=0.1):
    rows, cols, _ = img.shape
    j_width = float(cols) * random.uniform(1 - jitter, 1 + jitter)
    j_height = float(rows) * random.uniform(1 - jitter, 1 + jitter)
    img = cv2.resize(img, (int(j_width), int(j_height)))
    return img

def rotate(img, angle=5):

    scale = random.uniform(0.9, 1.1)
    angle = random.uniform(-angle, angle)

    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
    dst = img.copy()
    dst = cv2.warpAffine(img, M, (cols, rows), dst, cv2.INTER_LINEAR)

    return dst

def perspective(img):

    h, w, _ = img.shape
    per = random.uniform(0.05, 0.1)
    w_p = int(w * per)
    h_p = int(h * per)

    pts1 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
    pts2 = np.float32([[random.randint(0, w_p), random.randint(0, h_p)],
                       [random.randint(0, w_p), h - random.randint(0, h_p)],
                       [w - random.randint(0, w_p), random.randint(0, h_p)],
                       [w - random.randint(0, w_p), h - random.randint(0, h_p)]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (w, h))
    return img

def crop_subimage(img, margin=3):
    ran_margin = random.randint(0, margin)
    rows, cols, _ = img.shape
    crop_h = rows - ran_margin
    crop_w = cols - ran_margin
    row_start = random.randint(0, ran_margin)
    cols_start = random.randint(0, ran_margin)
    sub_img = img[row_start:row_start + crop_h, cols_start:cols_start + crop_w]
    return sub_img

def hsv_space_variation(ori_img, scale):

    rows, cols, _ = ori_img.shape

    hsv_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2HSV)
    hsv_img = np.array(hsv_img, dtype=np.float32)
    img = hsv_img[:, :, 2]

    # gau noise
    noise_std = random.randint(5, 20)
    noise = np.random.normal(0, noise_std, (rows, cols))

    # brightness scale
    img = img * scale
    img = np.clip(img, 0, 255)
    img = np.add(img, noise)

    # random hue variation
    hsv_img[:, :, 0] += random.randint(-5, 5)

    # random sat variation
    hsv_img[:, :, 1] += random.randint(-30, 30)

    hsv_img[:, :, 2] = img
    hsv_img = np.clip(hsv_img, 0, 255)
    hsv_img = np.array(hsv_img, dtype=np.uint8)
    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    return rgb_img

def data_augmentation(img):

    img = jitter(img)

    if random.choice([True, False]):
        img = rotate(img)

    if random.choice([True, False]):
        img = perspective(img)

    img = crop_subimage(img)

    bright_scale = random.uniform(0.6, 1.2)
    img_out = hsv_space_variation(img, scale=bright_scale)

    return img_out