from django.urls import path
from .views import login_view, verify_otp, upload_mri

urlpatterns = [
    path('', login_view, name='login'),
    path('verify/', verify_otp, name='verify'),
    path('upload/', upload_mri, name='upload'),
]