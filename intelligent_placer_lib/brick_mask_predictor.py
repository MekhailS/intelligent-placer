import json
import os

import numpy as np
import onnxruntime

from intelligent_placer_lib.utils import load_image


class BrickMaskPredictor:
    PATH_MODEL = "model/model.onnx"
    PATH_MASKS_MAPPING = "perfect_masks/id_to_mask.json"
    PATH_MASKS = "perfect_masks"

    def __init__(self):
        self.ort_session = onnxruntime.InferenceSession(BrickMaskPredictor.PATH_MODEL)
        with open(BrickMaskPredictor.PATH_MASKS_MAPPING, "r") as file:
            self.perfect_mask_mapping = json.load(file)

    def predict_perfect_masks(self, crops):
        predictions = self._predict_bricks_ids(crops)

        perfect_masks = []
        for prediction in predictions:
            mask_path = os.path.join(BrickMaskPredictor.PATH_MASKS, self.perfect_mask_mapping[str(prediction)])
            perfect_masks.append(load_image(mask_path))

        return perfect_masks

    def _predict_bricks_ids(self, crops):
        crops_in_batch = np.asarray(crops).transpose((0, 3, 1, 2)).astype(np.float32) / 255.0

        ort_inputs = {self.ort_session.get_inputs()[0].name: crops_in_batch}
        ort_outs = self.ort_session.run(None, ort_inputs)
        predictions = ort_outs[0].argmax(axis=1)
        return predictions
