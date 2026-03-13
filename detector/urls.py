from django.urls import path
from . import views

urlpatterns = [

    path("", views.login_view, name="login"),

    path("verify/", views.verify_view, name="verify"),

    path("upload/", views.upload_page, name="upload"),

    path("predict/", views.predict_mri, name="predict"),
    
    path("resend-otp/", views.resend_otp, name="resend_otp"),
    
    path("logout/", views.logout_view, name="logout"),

]