import cv2
import os

from collect_images import VideoCaptureLoop


def main():
    FILE_LAST_CAPTURE_ID = r'../data/state/last_capture_id.txt'
    PATH_FIT_IMAGES = r'../examples/fit'
    PATH_NOT_FIT_IMAGES = r'../examples/not_fit'

    with open(FILE_LAST_CAPTURE_ID) as file:
        cur_capture_id = int(file.readline())

    is_fit_mode = True

    def change_cur_mode(_):
        nonlocal is_fit_mode

        is_fit_mode = not is_fit_mode
        new_mode = "fit" if is_fit_mode else "not fit"
        print(f"set mode to {new_mode}")

        return True

    def capture_img_in_some_mode(img):
        nonlocal cur_capture_id

        path = PATH_FIT_IMAGES if is_fit_mode else PATH_NOT_FIT_IMAGES
        mode = "fit" if is_fit_mode else "not it"

        cur_capture_id += 1
        img_path = os.path.join(path, f'{cur_capture_id}_{mode}.png')
        cv2.imwrite(img_path, img)
        return True

    video_capture_loop = VideoCaptureLoop()

    video_capture_loop.add_callback(' ', change_cur_mode)
    video_capture_loop.add_callback('\r', capture_img_in_some_mode)

    video_capture_loop.start()

    pass


if __name__ == '__main__':
    main()
