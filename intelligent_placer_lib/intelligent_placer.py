from intelligent_placer_lib.brick_mask_predictor import BrickMaskPredictor
from intelligent_placer_lib.brick_on_placer_field import BrickOnPlacerField
from intelligent_placer_lib.area_bricks_fitter import AreaBricksFitter
from intelligent_placer_lib.papers_splitter import PapersSplitter
from intelligent_placer_lib.utils import load_image


def check_image(path_to_image):
    img = load_image(path_to_image)
    papers_splitter = PapersSplitter(img)
    papers_splitter.launch_processing()

    brick_mask_predictor = BrickMaskPredictor()
    objects_perfect_masks = brick_mask_predictor.predict_perfect_masks(papers_splitter.object_crops)

    area_to_fit_in = papers_splitter.area_to_fit
    bricks = [BrickOnPlacerField.create_with_mask_preprocessing(mask) for mask in objects_perfect_masks]

    area_bricks_fitter = AreaBricksFitter(area_to_fit_in, bricks)
    return area_bricks_fitter.fit()
