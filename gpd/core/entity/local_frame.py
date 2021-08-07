import numpy as np
import open3d as o3d


class LocalFrame:
    """
    The representation of local reference frame
    """

    def __init__(self, sample_point):
        """
        Create the local reference frame
        :param sample_point: the point itself. np.ndarray, shape=(3,)
        """
        self.curvature_axis = None
        self.normal_axis = None
        self.binormal_axis = None
        # the local reference frame belongs to this point
        self.sample = sample_point.reshape(3, )

    def calculate_axis(self, normals: np.ndarray):
        """
        calculate all three axis for the local reference frame according to the normals of neighboring points
        :param normals: normals of neighboring points, shape=(num of neighbors, 3)
        :return:
        """
        # 1. calculate "curvature axis" (corresponding to minor principal curvature axis)
        M = normals.T @ normals
        eig_values, eig_vectors = np.linalg.eig(M)
        eig_values = np.real(eig_values)
        eig_vectors = np.real(eig_vectors)
        min_index = np.argmin(eig_values)
        self.curvature_axis = eig_vectors[:, min_index]

        # 2. calculate surface normal
        max_index = np.argmax(eig_values)
        self.normal_axis = eig_vectors[:, max_index]

        # 3. ensure that the new normal is pointing in the same direction as the existing normals.
        normals_avg = normals.sum(axis=0)
        normals_avg /= np.linalg.norm(normals_avg)  # shape=(3,)
        if normals_avg @ self.normal_axis < 0:
            self.normal_axis *= -1.0

        # 4. create binormal (corresponds to major principal curvature axis)
        self.binormal_axis = np.cross(self.curvature_axis, self.normal_axis)

    def __str__(self):
        return f'Sample = {self.sample}, ' \
               f'Curvature axis = {self.curvature_axis}, ' \
               f'Normal axis = {self.normal_axis}, ' \
               f'Binormal axis = {self.binormal_axis}'

    def check_axis_orthogonal(self):
        print('curvature_axis @ normal_axis', self.curvature_axis @ self.normal_axis)
        print('curvature_axis @ binormal_axis', self.curvature_axis @ self.binormal_axis)
        print('normal_axis @ binormal_axis', self.normal_axis @ self.binormal_axis)
        print('-' * 20)
