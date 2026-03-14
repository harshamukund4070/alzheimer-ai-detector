from django.urls import path
from . import views

urlpatterns = [

    path("", views.login_view, name="login"),
    path("signup/", views.signup_view, name="signup"),
    path("social-login/<str:provider>/", views.social_login, name="social_login"),
    path("forgot-password/", views.forgot_password_view, name="forgot_password"),
    path("verify/", views.verify_view, name="verify"),
    path("upload/", views.upload_page, name="upload"),
    path("predict/", views.predict_mri, name="predict"),
    path("report/<int:record_id>/", views.download_report, name="download_report"),
    path("performance/", views.performance_page, name="performance"),
    path("info/", views.info_page, name="info"),
    path("resend-otp/", views.resend_otp, name="resend_otp"),
    path("logout/", views.logout_view, name="logout"),

]