from copy import copy

import cv2
import numpy as np
from scipy import ndimage


class BrickOnPlacerField:
    @staticmethod
    def create_with_mask_preprocessing(mask, initial_position=None, initial_rotation=0):
        if initial_position is None:
            initial_position = [0, 0]

        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        mask = mask[:, :, 0]
        mask = mask / 255
        return BrickOnPlacerField(mask, initial_position, initial_rotation)

    def __init__(self, mask, initial_position, initial_rotation):
        self.mask = mask
        initial_position = np.asarray(initial_position)
        self.positions = [initial_position]
        self.rotations = [initial_rotation]

        self.area = int(self.mask.shape[0] * self.mask.shape[1])

    def current_position(self):
        return np.sum(self.positions, axis=0)

    def current_rotation(self):
        return np.sum(self.rotations)

    def clone(self):
        cloned = BrickOnPlacerField(self.mask, None, None)
        cloned.positions = copy(self.positions)
        cloned.rotations = copy(self.rotations)
        return cloned

    def generate_same_position_new_rotation(self):
        res = self.clone()
        rotation_change = int(np.random.uniform(0, 360, 1)[0])
        res.positions.append(np.asarray([0, 0]))
        res.rotations.append(rotation_change)
        return res

    def generate_new_position_new_rotation(self):
        res = self.generate_same_position_new_rotation()
        position_change = np.random.normal(0, 50, 2).astype(int)
        res.positions[-1] = position_change
        return res

    def generate_new_position_same_rotation(self):
        res = self.generate_new_position_new_rotation()
        res.positions[-1] = np.asarray([0, 0])
        return res

    def rotate_mask(self):
        mask_rotated = ndimage.rotate(self.mask, self.current_rotation(), reshape=True)
        mask_rotated[mask_rotated > 0.5] = 1
        mask_rotated[mask_rotated <= 0.5] = 0
        mask_rotated = mask_rotated.astype(int)
        return mask_rotated

    def with_rotation(self, rotation):
        res = self.clone()
        res.rotations.append(rotation)
        res.positions.append(np.asarray([0, 0]))
        return res

    def with_position(self, position):
        res = self.clone()
        res.positions.append(np.asarray(position))
        res.rotations.append(0)
        return res

    def apply_on_field(self, placer_field):
        final_mask = self.rotate_mask()

        h, w = final_mask.shape
        pos_x, pos_y = self.current_position() - np.asarray([w // 2, h // 2])
        try:
            placer_field[pos_y:pos_y + h, pos_x:pos_x + w] += final_mask
        except:
            placer_field[:, :] = 999999