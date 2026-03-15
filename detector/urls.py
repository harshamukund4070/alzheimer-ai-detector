from django.urls import path
from . import views

urlpatterns = [

    # ── Auth ──────────────────────────────────────────────────────────────
    path("", views.login_view, name="login"),
    path("signup/", views.signup_view, name="signup"),
    path("forgot-password/", views.forgot_password_view, name="forgot_password"),
    path("verify/", views.verify_view, name="verify"),
    path("resend-otp/", views.resend_otp, name="resend_otp"),
    path("logout/", views.logout_view, name="logout"),

    # ── Google OAuth (real flow) ───────────────────────────────────────────
    path("google-login/", views.google_login, name="google_login"),
    path("google-callback/", views.google_callback, name="google_callback"),

    # ── Dashboard & Features ───────────────────────────────────────────────
    path("upload/", views.upload_page, name="upload"),
    path("predict/", views.predict_mri, name="predict"),
    path("report/<int:record_id>/", views.download_report, name="download_report"),
    path("performance/", views.performance_page, name="performance"),
    path("info/", views.info_page, name="info"),
    path("helpcentre/", views.helpcentre_page, name="helpcentre"),

]