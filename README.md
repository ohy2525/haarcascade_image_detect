# haarcascade_image_detect
01 - google search api를 통한 사진 다운로드\
02 - 확장자를 .jpg로 바꿔주는 코드\
03 - 데이터 증강 및 저장\
04 - haarcascade_eye를 통해 눈을 검출하고 검출한 이미지를 저장\
05 - 쌍커풀과 무쌍을 구분하는 모델 생성\
06 - 새로운 이미지로 확인


### Django

 1.Django porject 생성

(cmd에서)django-admin startproject mysite


2. mysite/settings.py 수정 \
import os\
TIME_ZONE = 'Asia/Seoul'\
LANGUAGE_CODE = 'en-us'\
STATIC_URL = '/static/'\
STATIC_ROOT = os.path.join(BASE_DIR, 'static')\

3. Migration\
python manage.py migrate\

4. Web server 실행\
python manage.py runserver\
