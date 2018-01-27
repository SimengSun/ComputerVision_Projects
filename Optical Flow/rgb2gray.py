def rgb2gray(img):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    img_gray = 0.2989 * r + 0.5870 * g, 0.1140 * b
    return img_gray
