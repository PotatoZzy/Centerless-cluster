import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder


def extract_features(image_path, target_dim=728):
    """
    Extract combined features from image to get 728 dimensions vector
    """
    # Read the image and binarize
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    features = []

    # 1. Zernike moments (49 dimensions)
    try:
        zernike = cv2.ZernikeMoments(binary, radius=128, degree=12)
        features.extend(zernike)
    except:
        features.extend([0] * 49)

    # 2. Hu moments (7 dimensions)
    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).flatten()
    features.extend(hu)

    # 3. HOG Features (512 dimensions)
    resized = cv2.resize(binary, (128, 128))
    hog_feat = hog(resized, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2),
                   orientations=8,
                   feature_vector=True)
    features.extend(hog_feat)

    # 4. Fourier Descriptors from contour (160 dimensions)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        fourier = fourier_descriptor(cnt)
        features.extend(fourier)
    else:
        features.extend([0] * 160)

    return np.array(features[:target_dim]) if len(features) >= target_dim else \
        np.pad(features, (0, target_dim - len(features)), 'constant')


def fourier_descriptor(contour, num_descriptors=80):
    complex_coords = contour[:, 0, 0] + 1j * contour[:, 0, 1]
    descriptors = np.fft.fft(complex_coords)
    return np.concatenate([descriptors.real[:num_descriptors],
                           descriptors.imag[:num_descriptors]])


def process_mpeg7(folder_path):
    all_features = []
    all_labels = []

    class_names = sorted(set(
        filename.split('-')[0]
        for filename in os.listdir(folder_path)
        if filename.endswith('.png')
    ))

    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png'):
            class_name = filename.split('-')[0]

            img_path = os.path.join(folder_path, filename)
            features = extract_features(img_path)

            all_features.append(features)
            all_labels.append(class_to_idx[class_name])

    X = np.vstack(all_features)
    y = np.array(all_labels)

    return X, y, class_names


if __name__ == "__main__":
    current_path = os.path.dirname(__file__)
    dataset_folder = os.path.join(current_path, 'data', 'mpeg7')

    X, y_true, classes = process_mpeg7(dataset_folder)


    print(f"X.shape: {X.shape}")  # (n_samples, 728)
    print(f"y.shape: {y_true.shape}")  # (n_samples,)
    print(f"class: {classes}")
