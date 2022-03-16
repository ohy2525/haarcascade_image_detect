#-*- coding: utf-8 -*- 
import os
import pathlib
import glob
import cv2
import settings

def load_name_images(image_path_pattern):
    name_images = []
    # 지정한 Path Pattern에 일치하는 파일 얻기
    image_paths = glob.glob(image_path_pattern)
    # 파일별로 읽기
    for image_path in image_paths:
        # 파일 경로
        path = pathlib.Path(image_path)
        fullpath = str(path.resolve())
        print(f"이미지 파일 절대경로:{fullpath}")
        # 파일명
        filename = path.name
        # 이미지 읽기
        image = cv2.imread(fullpath)
        if image is None :
            print(f'이미지 파일을 읽을 수 없습니다')
            continue
        # TO-DO 
        name_images.append((filename, image))
        
    return name_images

def detect_image_eye(file_path, image, cascade_filepath):
    # 이미지 파일의 Grayscale화
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 캐스케이드 파일 읽기
    cascade = cv2.CascadeClassifier(cascade_filepath)
    # 눈인식
    eyes = cascade.detectMultiScale(image_gs, scaleFactor=1.01, minNeighbors=20, minSize=(64,64))
    if len(eyes) == 0:
        print(f"눈인식실패")
        return
    # TO-DO
    # 1개 이상의 눈인식
    eye_count = 1
    for (xpos, ypos, width, height) in eyes :
        eye_image = image[ypos:ypos + height, xpos:xpos + width]
        if eye_image.shape[0] > 64:
            eye_image = cv2.resize(eye_image, (64,64))

        path = pathlib.Path(file_path)    
        directory = str(path.parent.resolve())#부모 경로
        filename = path.stem  #확장자 빼고
        extension = path.suffix  #확장자
        output_path = os.path.join(directory, f"{filename}_{eye_count:03}{extension}")
        print(f"출력파일(절대경로):{output_path}")
        try:
            cv2.imwrite(output_path, eye_image)
        except :
            print('Exception occured:{}, {}'.format(output_path, eye_image))
        eye_count += 1
        

    # TO-DO 

def delete_dir(dir_path, is_delete_top_dir=True):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    if is_delete_top_dir:
        os.rmdir(dir_path)

RETURN_SUCCESS = 0
RETURN_FAILURE = -1
# Origin Image Pattern
IMAGE_PATH_PATTERN = "./origin_image/*"
# Output Directory
OUTPUT_IMAGE_DIR = "./eye_image"



def main():
    print("===================================================================")
    print("이미지 눈인식 OpenCV 이용")
    print("지정한 이미지 파일의 정면눈을 인식하고, 64x64 사이즈로 변경")
    print("===================================================================")

    # 디렉토리 작성
    if not os.path.isdir(OUTPUT_IMAGE_DIR):
        os.mkdir(OUTPUT_IMAGE_DIR)
    # 디렉토리 내의 파일 제거
    delete_dir(OUTPUT_IMAGE_DIR, False)

    # 이미지 파일 읽기
    # TO-DO 

    name_images = load_name_images(IMAGE_PATH_PATTERN)

    # 이미지별로 눈인식
    for name_image in name_images:
        # TO-DO 
        file_path = os.path.join(OUTPUT_IMAGE_DIR, f"{name_image[0]}")
        image = name_image[1]
        cascade_filepath = settings.CASCADE_FILE_PATH
        detect_image_eye(file_path, image, cascade_filepath)
    return RETURN_SUCCESS

if __name__ == "__main__":
    main()