from django.shortcuts import render
from django.views.generic import TemplateView
from .forms import ImageForm
from .main import detect

class PredView(TemplateView):
    # 생성자
    def __init__(self):
        self.params = {'result_list': [],
                       'result_name': "",
                       'result_img': "",
                       'form': ImageForm()}

    # GET request (index.html를 초기표시)
    def get(self, req):
        return render(req, 'pred/index.html', self.params)

    # POST request (index.html에 결과를 표시)
    def post(self, req):
        # POST 된 Form 데이터 얻기
        form = ImageForm(req.POST, req.FILES)
        # Form data 에러 체크
        if not form.is_valid():
            raise ValueError('invalid form')
        # Form data로 부터 이미지 파일 얻기
        image = form.cleaned_data['image']
        # 이미지 파일을 지정하여 얼굴 분류
        result = detect(image)
        # 얼굴 분류의 결과 저장
        self.params['result_list'], self.params['result_name'], self.params['result_img'] = result
        # 페이지에 화면 표시
        return render(req, 'pred/index.html', self.params)