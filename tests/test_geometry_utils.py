import numpy as np

from random_matrix.utils import geometry_utils


def test_minkowski_sum():
    
    one = np.array([[-4,2],[-3,1],[-2,2]])
    two = np.array([[2,1],[4,1],[4,3],[2,3]])
    s = geometry_utils.minkowski_sum(one, two) 
    print(s)   
    print("---")
    
    polygon_one = np.array([[-1, -1], [1, -1], [0, 1]])
    polygon_two = np.array([[3, -1], [5, -1], [5, 1], [3, 1]])
    s = geometry_utils.minkowski_sum(polygon_one, polygon_two)
    print(s)
    print("---")

    one = np.array([[0,0],[1,0],[1,1],[0,1]])
    two = np.array([[1,1],[2,1],[2,2],[1,2]])
    s = geometry_utils.minkowski_sum(one, two)
    print(s)
    print("----")
    
test_minkowski_sum()

