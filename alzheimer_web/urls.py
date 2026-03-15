from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve
import os

STATIC_ROOT_DIR = os.path.join(settings.BASE_DIR, 'staticfiles')

urlpatterns = [
    path("", include("detector.urls")),
    # PWA files served directly
    path("manifest.json", serve, {"document_root": STATIC_ROOT_DIR, "path": "manifest.json"}),
    path("sw.js", serve, {"document_root": STATIC_ROOT_DIR, "path": "sw.js"}),
    path("icon-192.png", serve, {"document_root": STATIC_ROOT_DIR, "path": "icon-192.png"}),
    path("icon-512.png", serve, {"document_root": STATIC_ROOT_DIR, "path": "icon-512.png"}),
]

# Serve uploaded media files (MRI images)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)