from django.urls import path
from detector.views import login_view, verify_view, upload_mri

urlpatterns = [

    path("", login_view, name="login"),
    path("verify/", verify_view, name="verify"),
    path("upload/", upload_mri, name="upload"),

]