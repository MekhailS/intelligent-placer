import cv2
import json
import os


class VideoCaptureLoop:

    def __init__(self):
        self.video = cv2.VideoCapture(2)
        self.callback_dict = {}
        self.add_callback('q', lambda x: False)

    def add_callback(self, character, func):
        self.callback_dict[ord(character)] = func

    def start(self):
        while True:
            ret, frame = self.video.read()

            cv2.imshow('frame', frame)

            key = cv2.waitKey(1) & 0xFF

            if key in self.callback_dict:
                if not self.callback_dict[key](frame):
                    break


def main():
    FILE_LAST_CAPTURE_ID = r'../data/state/last_capture_id.txt'
    PATH_CAPTURES = r'../data/images'
    DATASET_JSON = r'../data/description/data.json'

    with open(FILE_LAST_CAPTURE_ID) as file:
        cur_capture_id = int(file.readline())

    with open(DATASET_JSON) as file:
        data = json.load(file)

    cur_brick_name = ''
    cur_brick_id = ''
    cur_brick_additional_info = ""

    def set_cur_brick(_):
        nonlocal cur_brick_name
        nonlocal cur_brick_id
        nonlocal cur_brick_additional_info

        cur_brick_id = str(input('enter current brick id: '))
        cur_brick_name = str(input('enter current brick name: '))
        cur_brick_additional_info = str(input('enter current brick additional info: '))
        return True

    def capture_frame(img):
        nonlocal cur_capture_id

        cur_capture_id += 1
        img_path = os.path.join(PATH_CAPTURES, f'{cur_capture_id}_{cur_brick_id}_{cur_brick_name}.png')
        cv2.imwrite(img_path, img)

        data['captures'].append(
            {
                "id": cur_brick_id,
                "name": cur_brick_name,
                "image_path": img_path,
                "additional_info": cur_brick_additional_info
            }
        )
        return True

    video_capture_loop = VideoCaptureLoop()

    video_capture_loop.add_callback(' ', set_cur_brick)
    video_capture_loop.add_callback('\r', capture_frame)

    video_capture_loop.start()

    with open(FILE_LAST_CAPTURE_ID, 'w') as file:
        file.write(str(cur_capture_id))

    with open(DATASET_JSON, 'w') as file:
        json.dump(data, file, indent=4)

    pass


if __name__ == '__main__':
    main()
