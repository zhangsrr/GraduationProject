from functools import partial
from typing import Set
import numpy as np
from scipy import stats
from pyknotid.spacecurves.spacecurve import SpaceCurve


class StreamSpaceCurve(SpaceCurve):
    def __init__(self, points, velocity):
        self.velocity = velocity
        super().__init__(points)
        self.arclengths = None
        self.segment_arclengths = None
    
    @staticmethod
    def get_direct_distance(a: np.array, b: np.array) -> int:
        return np.linalg.norm(a - b)

    @staticmethod
    def get_neighbors(array, center: int, radius: int) -> np.array:
        start = max(0, center - radius)
        end = min(array.shape[0] - 1, center + radius)
        neighbors = array[start:end]
        if neighbors.shape[0] == 0:
            raise ('Invalid neighborhood for `{neighbors.shape}`: ({center}, {radius})')
        return neighbors

    def get_dirct_length(self, start_point : np.array = None):
        if start_point is None:
            start_point = self.points[0]
        dirct_lengths = np.array(list(map(partial(StreamSpaceCurve.get_direct_distance, start_point), self.points)))
        return dirct_lengths

    def get_curvature(self):
        curvature = super().curvatures()
        return curvature

    def get_torsion(self):
        torsion = super().torsions()
        torsion = np.nan_to_num(torsion)
        self.check_is_finite(torsion)
        return torsion

    def get_tortuosity(self):
        start_point = self.points[0]
        self.segment_arclengths = super().segment_arclengths()[1:]
        self.arclengths = np.cumsum(self.segment_arclengths)
        dirct_lengths = self.get_dirct_length()
        tortuosity = np.hstack(([1], np.divide(self.arclengths, dirct_lengths[1:])))
        return tortuosity

    def get_tangent(self):
        return np.tan(self.points)

    @staticmethod
    def find_segments(segments: np.array, value) -> Set:
        result = set()
        for i, segment in enumerate(segments):
            if not segment[0] <= value < segment[1]:
                result.add(i)
        return result
    
    @staticmethod
    def check_is_finite(data_array: np.array) -> bool:
        isfinite = np.isfinite(data_array)
        if not isfinite.all():
            error_index = np.nonzero(isfinite == False)
            error_values = data_array.take(list(zip(*error_index)))
            raise ('Infinite values in data_array `{error_values}` at {error_index}.')
        return isfinite
    
    def cart2sph(self, xyz: np.array):
        n, d = xyz.shape
        if d != 3:
            raise ('Invalid xyz shape `{xyz.shape}`.')
        self.check_is_finite(xyz)
        sphs = np.zeros((n, 2))
        xy = xyz[:, 0]**2 + xyz[:, 1]**2
        sphs[:, 0] = np.arctan2(np.sqrt(xy), xyz[:,2]) + np.pi # for elevation angle defined from Z-axis down
        #sphs[:, 0] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        try:
            sphs[:, 1] = np.arctan(xyz[:, 1] / xyz[:, 0]) + np.pi / 2
        except FloatingPointError as e:
            if e.args != ('invalid value encountered in true_divide',) and e.args != ('divide by zero encountered in true_divide',):
                raise e
            else:
                with np.errstate(all='ignore'):
                    sphs[:, 1] = np.arctan(xyz[:, 1] / xyz[:, 0]) + np.pi / 2
        sphs = np.nan_to_num(sphs)
        self.check_is_finite(sphs)
        return sphs
    
    def find_region_in_radians(self, spherical: np.array) -> bool:
        dimention, n, region_dimention = self.region_map.shape
        if region_dimention != 2:
            raise ('Invalid region_map {self.region_map.shape}')
        if dimention != spherical.shape[0]:
            raise ('Invalid spherical `{spherical}` for region_map {self.region_map.shape}')
        if not (0 <= spherical[0] <= 2 * np.pi and 0 <= spherical[1] <= np.pi):
            raise ('Invalid spherical: {spherical}')
        if spherical[1] == 0:
            return 0
        else:
            dismissed = set()
            whole_index_set = set(range(n))
            for i, segments in enumerate(self.region_map):
                dismissed |= self.find_segments(segments, spherical[i])
            result = whole_index_set - dismissed
            if len(result) != 1:
                raise ('Cannot find region for spherical `{spherical}`: {result}')
            return result.pop()
    
    def find_region_in_cartesians(self, cartesians: np.array) -> bool:
        sphs = self.cart2sph(cartesians)
        return np.apply_along_axis(self.find_region_in_radians, 1, sphs)
    
    def get_region(self):
        tangents = self.get_tangent()
        self.check_is_finite(tangents)
        # np.savetxt(str(Path(self.data_path, f'tangents.txt')), tangents)
        regions = self.find_region_in_cartesians(tangents).astype(int)
        return regions

    def get_velocity_direction_entropy(self):
        # NEIGHBORHOOD_RADIUS = 10
        # region_counter = self.region_map.shape[1]
        # regions = self.get_region()
        # entropy_list = []
        # for i in range(self.points.shape[0]):
        #     neighbors = self.get_neighbors(regions, i, NEIGHBORHOOD_RADIUS)
        #     distribution = np.bincount(neighbors)
        #     missing_length = region_counter - distribution.shape[0]
        #     padded_distribution = np.pad(distribution, (0, missing_length), 'constant', constant_values=0)
        #     entropy_list.append(stats.entropy(padded_distribution))
        #     # np.savetxt(str(Path(velocity_distribution_path, f'{i}.txt')), padded_distribution, fmt="%d")
        # velocity_direction_entropy = np.array(entropy_list)
        
        entropy_list = []
        for i in range(self.points.shape[0]):
            entropy_list.append(0)
        velocity_direction_entropy = np.array(entropy_list)
        
        return velocity_direction_entropy

    def get_point_features(self):
        curvature = self.get_curvature()
        torsion = self.get_torsion()
        tortuosity = self.get_tortuosity()
        velocity_direction_entropy = self.get_velocity_direction_entropy()
        features = np.stack([curvature, torsion, tortuosity, velocity_direction_entropy]).transpose()
        return features

    def get_total_length(self) -> int:
        total_length = self.arclength()
        if total_length == 0:
            raise ('Total length is 0.')
        return total_length
    
    def get_arc_length_matrix_array(self):
        return self.segment_arclengths
