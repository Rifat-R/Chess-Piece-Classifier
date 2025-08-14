from typing import List
from utils import utils

import numpy as np

N_DIMENSIONS = 10
K_VALUE = 5


def euclidean_distance_vectorized(test_point, train_points):
    # Use NumPy broadcasting to calculate the Euclidean distance for all points in train_points
    # Before, I was using a for loop to calculate the distance for each point in train_points which was extremely slow
    distances = np.sqrt(np.sum((train_points - test_point) ** 2, axis=1))
    return distances


def knn_algorithm(train_fvectors, train_labels, test_fvectors, k=K_VALUE):
    predictions = []
    k_neighbor_labels = []

    for test_point in test_fvectors:
        # Calculate distances between the test point and all training points
        distances = euclidean_distance_vectorized(test_point, train_fvectors)

        # Get the k smallest distances indices (default is K_VALUE)
        k_neighbors_indices = np.argsort(distances)[:k]

        # Get labels of k-nearest neighbors
        k_neighbors_labels = [train_labels[i] for i in k_neighbors_indices]

        # Predict the label by majority voting
        predicted_label = max(set(k_neighbors_labels), key=k_neighbors_labels.count)
        predictions.append(predicted_label)
        k_neighbor_labels.append(k_neighbors_labels)

    return predictions, k_neighbor_labels


def classify(
    train: np.ndarray, train_labels: np.ndarray, test: np.ndarray
) -> List[str]:
    """Classify a set of feature vectors using a training set.

    This implementation uses a k-Nearest Neighbors (KNN) classifier with Euclidean distance.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the predicted labels for each square.
    """
    predictions, _ = knn_algorithm(train, train_labels, test)
    return predictions


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors using PCA.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    pca_model = model["pca_model"]
    mean_vec = np.array(pca_model["mean_vec"])
    selected_eigenvectors = np.array(pca_model["selected_eigenvectors"])

    # Mean centering
    centered_data = data - mean_vec

    # Project data onto the new subspace
    reduced_data = np.dot(centered_data, selected_eigenvectors)

    return reduced_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """
    # Perform PCA for dimensionality reduction
    mean_vec: np.ndarray = np.mean(fvectors_train, axis=0)
    centered_data = fvectors_train - mean_vec
    cov_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    # Selects the first N_DIMENSIONS eigenvectors (This will be our feature vector space)
    selected_eigenvectors = sorted_eigenvectors[:, :N_DIMENSIONS]

    # Store PCA information in the model dictionary
    model = {}
    model["pca_model"] = {
        "mean_vec": mean_vec.tolist(),
        "selected_eigenvectors": selected_eigenvectors.tolist(),
    }
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    model["labels_train"] = labels_train.tolist()

    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D `feature` vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    labels = classify(fvectors_train, labels_train, fvectors_test)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    train_labels = np.array(model["labels_train"])
    train_fvectors = np.array(model["fvectors_train"])

    predictions, k_neighbor_labels = knn_algorithm(
        train_fvectors, train_labels, fvectors_test
    )

    # Put predictions into a 2d list to create boards of 64 squares
    boards = []
    for i in range(0, len(predictions), 64):
        boards.append(predictions[i : i + 64])

    for index, board in enumerate(boards):
        # If there is a pawn in the first or last row, then illegal board
        if "p" in board[:8] or "P" in board[-8:]:
            square_index = board.index("p") if "p" in board else board.index("P")
            actual_index = square_index + index * 64
            new_labels = k_neighbor_labels[actual_index]
            for label in new_labels:
                if label != "p" and label != "P":
                    # Replace the pawn with the new label
                    predictions[actual_index] = label
                    break

    return predictions
