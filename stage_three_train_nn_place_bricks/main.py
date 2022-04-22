from bricks_dataset import BrickDataset
import cv2

if __name__ == '__main__':
    brick_dataset = BrickDataset("../data/description/test.csv", "", transform=None)
    cv2.imshow('image', brick_dataset[len(brick_dataset) - 2][0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
