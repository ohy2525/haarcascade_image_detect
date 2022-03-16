from ast import NamedExpr
import sys
import os
import cv2
from cv2 import IMWRITE_PNG_STRATEGY_FILTERED
import keras
import numpy as np
import matplotlib.pyplot as plt
import settings

def detect_eye(model, cascade_filepath, image):
    # 이미지를 BGR형식에서 RGB형식으로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.imshow(image)
    # plt.show()
    # print(image.shape)

    # 그레이스케일 이미지로 변환
    image_gs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 눈인식 실행
    cascade = cv2.CascadeClassifier(cascade_filepath)
    eyes = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=15, minSize=(64,64))

    # 눈이 1개 이상 검출된 경우
    if len(eyes) > 0 :
        print(f"인식한 눈의 수 : {len(eyes)}")
        for (xpos, ypos, width, height) in eyes:
            #인식된 눈 자르기
            eye_image = image[ypos:ypos + height , xpos :xpos + width ]
            print(f"인식한 눈의 사이즈 : {eye_image.shape}")
            #인식한 눈의 사이즈 축소
            if eye_image.shape[0] < 64 or eye_image.shape[1] < 64:
                print("인식한 눈의 사이즈가 너무 작습니다.")
                continue
            eye_image = cv2.resize(eye_image, (64,64))

            #인식한 눈 주변에 붉은색 사각형 표시
            image=cv2.rectangle(image, (xpos,ypos), (xpos+width, ypos+height), (255, 0, 0), thickness = 2)

            #차원 변경 (Input 0 of layer "sequential" is incompatible with the layer: expected shape=(None, 64, 64, 3), found shape=(32, 64, 3))
            eye_image = np.expand_dims(eye_image, axis = 0)
            #인식한 눈로부터 이름 가져오기
            name = detect_who(model, eye_image)
            #인식한 눈에 이름 표시
            cv2.putText(image, name, (xpos, ypos + height + 20), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0 ), 2)
    # 눈이 검출되지 않은 경우
    else :
        print('눈을 인식할 수 없습니다')
    
    return image

def detect_who(model, eye_image):
    # 예측
    name = ""
    result = model.predict(eye_image)
    print(f"예측결과 : {result}")
    
    print(f"쌍커풀일 가능성 : {result[0][0]*100:.3f}%")
    print(f"무쌍일 가능성 : {result[0][1]*100:.3f}%")
    #이름 반환
    name_number_label = np.argmax(result)
    if name_number_label == 0:
        name = 'double eyelid'
    elif name_number_label == 1:
        name = 'single eyelid'
    return name

RETURN_SUCCESS = 0
RETURN_FAILURE = -1
# Inpou Model Directory
INPUT_MODEL_PATH = "./model/model.h5"

def main():
    print("===================================================================")
    print("Keras를 이용한 눈인식")
    print("학습 모델과 지정한 이미지 파일을 기본으로 연예인 구분하기")
    print("===================================================================")

    # 인수 체크
    argvs = sys.argv
    if len(argvs) != 2 or not os.path.exists(argvs[1]):
        print('이미지 파일을 지정해 주세요')
        return RETURN_FAILURE
    image_file_path = argvs[1]

    # 이미지 파일 읽기
    image = cv2.imread(image_file_path)
    if image is None :
        print('이미지 파일을 읽을 수 없습니다.')

    # 모델 파일 읽기
    if not os.path.exists(INPUT_MODEL_PATH):
        print('이미지를 검출하기 위한 모델 파일이 없습니다.')
        return RETURN_FAILURE
    model = keras.models.load_model(INPUT_MODEL_PATH)


    # 눈인식
    cascade_filepath = settings.CASCADE_FILE_PATH
    result_image = detect_eye(model, cascade_filepath, image)
    plt.imshow(result_image)
    plt.show()


    return RETURN_SUCCESS

if __name__ == "__main__":
    main()