import numpy as np

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    
    Parameters:
        point1 (ndarray): Array containing the coordinates of the first point.
        point2 (ndarray): Array containing the coordinates of the second point.
    
    Returns:
        float: Euclidean distance between the two points.
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))