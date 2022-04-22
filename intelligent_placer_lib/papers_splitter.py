from copy import copy
import cv2
import numpy as np

from intelligent_placer_lib.utils import get_outer_contours_by_mask, get_mask_by_contour, apply_mask, process_image


class PapersSplitter:
    def __init__(self, image):
        self.image = image

        self.paper_mask_left = None
        self.paper_image_left = None

        self.paper_image_right = None
        self.paper_mask_right = None

        self.crops = None
        self.area_to_fit = None

    def launch_processing(self):
        mask = process_image(self.image, self._get_mask_processors())

        self.paper_mask_left, self.paper_mask_right = PapersSplitter._get_papers_masks(self.image, mask)
        self.paper_image_left, self.paper_image_right = [apply_mask(self.image, paper_mask) for paper_mask in
                                                         [self.paper_mask_left, self.paper_mask_right]]

        objects_mask = process_image(
            self.paper_image_right,
            PapersSplitter._get_right_paper_mask_processors(self.paper_mask_right)
        )
        self.crops = PapersSplitter._get_crops_with_objects(
            self.paper_image_right,
            get_outer_contours_by_mask(objects_mask)
        )
        self.area_to_fit = PapersSplitter._get_area_crop(self.paper_image_left, self.paper_mask_left)

    @staticmethod
    def _get_area_crop(image_paper_left, paper_mask_left):
        area_border_mask = process_image(image_paper_left, PapersSplitter._get_left_paper_processors(paper_mask_left))
        contours = get_outer_contours_by_mask(area_border_mask)
        contour = max(contours, key=lambda contour: cv2.contourArea(contour))

        all_area = np.zeros_like(area_border_mask)
        cv2.drawContours(all_area, [contour], -1, 255, thickness=cv2.FILLED)

        crop = cv2.boundingRect(contour)
        x, y, w, h = crop

        res = all_area[y:y + h, x:x + w]
        _, res = cv2.threshold(res, 254, 255, cv2.THRESH_BINARY)
        return res

    @staticmethod
    def _get_mask_processors():
        return [
            lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            lambda img: cv2.medianBlur(img, ksize=7),
            lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            lambda img: PapersSplitter._postprocess_mask(img),
            lambda img: cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((6, 6)))
        ]

    @staticmethod
    def _get_left_paper_processors(paper_mask_left):
        dilate_kernel = np.asarray([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0]
        ], dtype=np.uint8)

        return [
            lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            lambda img: cv2.medianBlur(img, ksize=7),
            lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
            lambda mask: apply_mask(mask, PapersSplitter._expand_background_mask(paper_mask_left)),
            lambda mask: cv2.dilate(mask, dilate_kernel, iterations=3)
        ]

    @staticmethod
    def _get_right_paper_mask_processors(paper_mask_right):
        def laplacian_edge(img):
            ddepth = cv2.CV_16S
            kernel_size = 5
            dst = cv2.Laplacian(img, ddepth, ksize=kernel_size)
            abs_dst = cv2.convertScaleAbs(dst)
            return abs_dst

        def laplacian_with_otsu(img):
            mask_laplacian = cv2.threshold(laplacian_edge(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            mask_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            res_mask = np.zeros_like(mask_otsu)
            res_mask[mask_laplacian != 0] = 255
            res_mask[mask_otsu != 0] = 255
            return res_mask

        return [
            lambda img: cv2.medianBlur(img, ksize=13),
            lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            lambda img: laplacian_with_otsu(img),
            lambda mask: apply_mask(mask, PapersSplitter._expand_background_mask(paper_mask_right)),
            lambda mask: cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel=np.ones((4, 4)),
                                          borderType=cv2.BORDER_REFLECT),
            lambda mask: cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((40, 40)))
        ]

    @staticmethod
    def _postprocess_mask(mask):
        mask_res = copy(mask)
        h, w = mask.shape

        last_clr = 255
        times_change = 0
        for x in range(0, w):
            if times_change == 2:
                mask_res[list(range(0, 15)), x:] = 255
                break

            cur_color = mask[10, x]
            if cur_color != last_clr:
                times_change += 1
                last_clr = cur_color
                continue

            last_clr = cur_color

        return mask_res

    @staticmethod
    def _get_papers_masks(image, mask):
        contours = get_outer_contours_by_mask(mask)
        assert len(contours) > 0
        if len(contours) == 1:
            return [get_mask_by_contour(image, contours[0])]

        papers_contours = list(sorted(contours, key=lambda contour: cv2.contourArea(contour))[-2:])
        paper_contours_from_left_to_right = list(
            sorted(papers_contours, key=lambda contour: cv2.boundingRect(contour)[0]))

        return tuple([get_mask_by_contour(image, contour) for contour in paper_contours_from_left_to_right])

    @staticmethod
    def _expand_background_mask(background_mask):
        background_mask = cv2.bitwise_not(background_mask)
        background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_DILATE, kernel=np.ones((30, 30)),
                                           borderType=cv2.BORDER_REFLECT)
        background_mask = cv2.bitwise_not(background_mask)
        return background_mask

    @staticmethod
    def _crop_image_to_bbox(image, bbox, shape_to_crop):
        x_bb, y_bb, w_bb, h_bb = bbox

        # transform x, y, w, h to coordinates of middle points, and w/2, h/2
        x_bb, y_bb, w_bb, h_bb = x_bb + w_bb // 2, y_bb + h_bb // 2, w_bb // 2, h_bb // 2

        H_img, W_img, _ = image.shape

        # x, y, w, h are the characteristics of crop region
        w, h = shape_to_crop[1] // 2, shape_to_crop[0] // 2

        def get_crop_region_mid_coord_along_axis(w, x_bb, w_bb, W_img):
            if w_bb > w:
                return x_bb, w_bb

            # mid x must be in [x_min, x_max]
            x_min = max(x_bb + w_bb - w, w)
            x_max = min(W_img - w, x_bb - w_bb + w)

            if x_min <= x_bb <= x_max:
                return x_bb, w

            return (x_min, w) if x_bb < x_min else (x_max, w)

        # middle coordinates of crop region
        x, w = get_crop_region_mid_coord_along_axis(w, x_bb, w_bb, W_img)
        y, h = get_crop_region_mid_coord_along_axis(h, y_bb, h_bb, H_img)

        # transform coordinates back
        x_bb, y_bb, w_bb, h_bb = x_bb - w_bb, y_bb - h_bb, 2 * w_bb, 2 * h_bb
        x, y, w, h = x - w, y - h, 2 * w, 2 * h

        # calculate end coordinates of regions (crop and bbox)
        x_end, y_end, x_bb_end, y_bb_end = x + w, y + h, x_bb + w_bb, y_bb + h_bb

        # fit bbox to cropped region
        def get_bbox_coords_along_axis(all_coords):
            all_coords = sorted(all_coords)
            return all_coords[1], all_coords[2]

        x_bb, x_bb_end = get_bbox_coords_along_axis([x, x_bb, x_bb_end, x_end])
        y_bb, y_bb_end = get_bbox_coords_along_axis([y, y_bb, y_bb_end, y_end])

        # update bounding box relative to cropped region
        x_bb, y_bb, w_bb, h_bb = x_bb - x, y_bb - y, x_bb_end - x_bb, y_bb_end - y_bb

        img_res = image[y:y + h, x:x + w]
        if w > shape_to_crop[1] or h > shape_to_crop[0]:
            img_res = cv2.resize(img_res, (shape_to_crop[1], shape_to_crop[0]))

        return img_res, (x_bb, y_bb, w_bb, h_bb)

    @staticmethod
    def _get_crops_with_objects(image, contours):
        assert len(contours) > 0

        crops = []
        for contour in contours:
            bbox_object = cv2.boundingRect(contour)
            crop, _ = PapersSplitter._crop_image_to_bbox(image, bbox_object, (256, 256))
            crops.append(crop)

        return crops