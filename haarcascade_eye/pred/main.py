import base64
import io
import cv2
import keras
import numpy as np
from PIL import Image
from keras.backend import tensorflow_backend as backend
#import tensorflow.keras.backend as backend -> tensorflow 2.0에서 변경
from django.conf import settings

def detect(upload_image):
    result_name = upload_image.name
    result_list = []
    result_img = ''

    # 설정에서 Cascade 파일의 경로 취득
    cascade_file_path = settings.CASCADE_FILE_PATH
    # 설정에서 모델 파일의 경로 취득
    model_file_path = settings.MODEL_FILE_PATH
    # Keras의 모델을 읽어오기
    model = keras.models.load_model(model_file_path)
    # 업로드된 이미지 파일을 메모리에서 OpenCV 이미지로 저장
    image = np.asarray(Image.open(upload_image))
    # 화상을 OpenCV BGR에서 RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 화상을 RGB에서 GRAY로 변환
    image_gs = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # Cascade 파일을 읽어오기
    cascade = cv2.CascadeClassifier(cascade_file_path)
    # OpenCV를 이용해 눈 인식
    face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1,
                                         minNeighbors=5, minSize=(64, 64))

    # 눈이 1개 이상인 경우
    if len(face_list) > 0:
        count = 1
        for (xpos, ypos, width, height) in face_list:
            # 인식한 눈을 잘라냄
            face_image = image_rgb[ypos:ypos+height, xpos:xpos+width]
            # 잘라낸 눈이 너무 작으면 스킵
            if face_image.shape[0] < 64 or face_image.shape[1] < 64:
                continue
            # 인식한 눈의 사이즈를 축소
            face_image = cv2.resize(face_image, (64, 64))
            # 인식한 눈 주변을 붉은 색으로 표시
            cv2.rectangle(image_rgb, (xpos, ypos),
                          (xpos+width, ypos+height), (0, 0, 255), thickness=2)
            # 인식한 눈을 1장의 화상 이미지로 합하는 배열로 변환
            face_image = np.expand_dims(face_image, axis=0)
            # 인식한 눈에서 이름을 추출
            name, result = detect_who(model, face_image)
            # 인식한 눈에 이름을 추가
            cv2.putText(image_rgb, f"{count}. {name}", (xpos, ypos+height+20),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            # 결과를 리스트에 저장
            result_list.append(result)
            count = count + 1

    # 화상을 PNG로 변환
    is_success, img_buffer = cv2.imencode(".png", image_rgb)
    if is_success:
        # 화상을 인메모리의 바이너리 스트림으로 전달
        io_buffer = io.BytesIO(img_buffer)
        # 인메모리의 바이너리 스트림에서 BASE64 인코드 변환
        result_img = base64.b64encode(io_buffer.getvalue()).decode().replace("'", "")

    # tensorflow의 백엔드 클리어
    backend.clear_session()
    # 결과 반환
    return (result_list, result_name, result_img)

def detect_who(model, face_image):
    # 예측
    predicted = model.predict(face_image)
    # 결과
    name = ""
    result = f"쌍커풀일 가능성:{predicted[0][0]*100:.3f}% / 무쌍일 가능성:{predicted[0][1]*100:.3f}%"
    name_number_label = np.argmax(predicted)
    if name_number_label == 0:
        name = "double eyelid"
    elif name_number_label == 1:
        name = "single eyelid"
    return (name, result)