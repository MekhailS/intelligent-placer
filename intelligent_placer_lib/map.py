from copy import copy

import cv2
import numpy as np
from scipy import ndimage

from intelligent_placer_lib.brick_on_field import BrickOnField
from intelligent_placer_lib.utils import get_outer_contours_by_mask

MAP_OUTSIDE_VALUE = -1000
OUT_OF_BORDER_COEF = 10


class Map:
    def __init__(
            self,
            area_crop,
            initial_bricks,
            grid_points_x=25,
            grid_points_y=25,
            num_rotations=18
    ):
        self.area_crop = area_crop
        self.grid_points_x = grid_points_x
        self.grid_points_y = grid_points_y
        self.num_rotations = num_rotations

        self.field = np.ones(np.asarray(area_crop.shape) * 2) * MAP_OUTSIDE_VALUE
        contour = get_outer_contours_by_mask(area_crop)[0]
        area = np.ones_like(area_crop) * MAP_OUTSIDE_VALUE
        cv2.fillPoly(area, [contour], 1)

        area_h, area_w = area_crop.shape
        pos_y, pos_x = np.asarray(self.field.shape) // 2 - np.asarray([it // 2 for it in area_crop.shape])
        self.field[pos_y:pos_y + area_h, pos_x:pos_x + area_w] = area

        self.grid_x = np.linspace(start=pos_x, stop=pos_x + area_w, endpoint=True, num=grid_points_x).astype(int)
        self.grid_y = np.linspace(start=pos_y, stop=pos_y + area_h, endpoint=True, num=grid_points_y).astype(int)

        self.rotations = np.linspace(start=0, stop=360, endpoint=False, num=num_rotations)

        self.initial_bricks = list(reversed(sorted(initial_bricks, key=lambda brick: brick.area)))
        self.optimal_bricks = []

    def rotated(self):
        rotated_crop = ndimage.rotate(self.area_crop, 90)
        return Map(rotated_crop, self.initial_bricks, self.grid_points_x, self.grid_points_y, self.num_rotations)

    def compute_field_with_optimal_bricks(self):
        res = copy(self.field)
        for brick in self.optimal_bricks:
            brick: BrickOnField
            brick.apply_on_field(res)

        return res

    def fit(self):
        def fit_one_brick(brick: BrickOnField):
            for x in self.grid_x:
                for y in self.grid_y:
                    if self.field[y, x] != 1:
                        continue

                    for rotation in self.rotations:
                        field_for_placing = self.compute_field_with_optimal_bricks()
                        brick_to_place = brick.with_position([x, y]).with_rotation(rotation)
                        brick_to_place.apply_on_field(field_for_placing)

                        loss = Map.get_loss_on_field(field_for_placing)
                        if loss == 0:
                            self.optimal_bricks.append(brick_to_place)
                            return True

            return False

        for brick in self.initial_bricks:
            if not fit_one_brick(brick):
                return False

        return True

    @staticmethod
    def get_loss_on_field(field):
        field = field.astype(int)
        loss_from_out_of_border = np.count_nonzero(
            np.logical_and(field > MAP_OUTSIDE_VALUE, field < 0)) * OUT_OF_BORDER_COEF
        loss_from_objects_intersection = np.sum(field * (field > 2))

        return loss_from_objects_intersection + loss_from_out_of_border

    @staticmethod
    def get_field_for_plot(field):
        field_to_plot = copy(field)
        field_to_plot[field_to_plot == MAP_OUTSIDE_VALUE] = 0
        field_to_plot[field_to_plot < 0] = 0

        d = np.max(field_to_plot)
        field_to_plot = field_to_plot / d * 255
        return field_to_plot.astype(int)
