import os
import pathlib
import glob
import cv2
import settings



def change_filename(image_path_pattern):
    image_paths = glob.glob(image_path_pattern)
    for image_path in image_paths:
        if not os.path.isdir(image_path):
            filename = os.path.splitext(image_path)
            os.rename(image_path, filename[0] + '.jpg')
            print('확장자를 변경하였습니다.')
        

RETURN_SUCCESS = 0
IMAGE_PATH_PATTERN = "./origin_image/*"


def main():
    print("===================================================================")
    print("파일 확장자를 .jpg로 변경")
    print("===================================================================")

    # 디렉토리 작성
    change_filename(IMAGE_PATH_PATTERN)
    return RETURN_SUCCESS


if __name__ == "__main__":
    main()