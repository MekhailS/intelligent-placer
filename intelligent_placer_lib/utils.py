import cv2
import matplotlib.pyplot as plt
from copy import copy
import numpy as np


def load_image(path: str):
  img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def plot_images(images: list):
  _, axs = plt.subplots(len(images), figsize=(16, 16*len(images)))
  print(axs)
  if len(images) == 1:
    axs.imshow(images[0], cmap="gray")
  else:
    axs = axs.flatten()
    for img, ax in zip(images, axs):
      ax.imshow(img, cmap="gray")
  plt.show()


def process_image(image, image_processors: list):
  for processor in image_processors:
    image = processor(image)

  return image


def process_image_and_plot_image_transformations(image, image_processors: list):
  image_transformations = [image]
  for processor in image_processors:
    image_transformations.append(processor(image_transformations[-1]))

  plot_images(image_transformations)


def get_outer_contours_by_mask(mask):
  contours, _ = cv2.findContours(copy(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = list(filter(lambda contour: cv2.contourArea(contour) > 200, contours))
  return contours


def get_bboxes_by_mask(mask):
  return [cv2.boundingRect(contour) for contour in get_outer_contours_by_mask(mask)]


def draw_contours(image, contours):
  image = copy(image)
  cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
  return image


def draw_bboxes(image, bboxes):
    image = copy(image)
    for i, (x, y, w, h) in enumerate(bboxes):
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(image, f'{i}', (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), thickness=5)

    return image


def get_mask_by_contour(image, contour):
  mask = np.zeros_like(image)
  cv2.fillPoly(mask, [contour], (255, 255, 255))
  return mask


def apply_mask(image, mask):
  mask_boolean = mask != (255, 255, 255)
  if image.ndim == 2:
    mask_channels = [mask_boolean[:, :, i] for i in (0, 1, 2)]
    mask_boolean = np.logical_or(*mask_channels)

  return np.where(mask_boolean, 0, image)
