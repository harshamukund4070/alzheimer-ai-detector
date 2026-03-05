import os
import random
import numpy as np
from PIL import Image

from django.shortcuts import render, redirect
from django.conf import settings
from django.core.mail import EmailMultiAlternatives

from tensorflow.keras.models import load_model


# -----------------------------
# MODEL PATH
# -----------------------------
MODEL_PATH = os.path.join(settings.BASE_DIR, "detector", "models", "alzheimer_model.h5")

model = None


# -----------------------------
# LOAD MODEL SAFELY
# -----------------------------
def get_model():
    global model
    if model is None:
        model = load_model(MODEL_PATH, compile=False)
    return model


# -----------------------------
# LOGIN VIEW (SEND OTP)
# -----------------------------
def login_view(request):

    if request.method == "POST":

        email = request.POST.get("email")

        otp = str(random.randint(100000, 999999))

        request.session["otp"] = otp
        request.session["email"] = email

        subject = "Harsha Pvt Limited – Your Login OTP"
        text_content = f"Your OTP is {otp}. Valid for 5 minutes."

        html_content = f"""
        <html>
        <body style="font-family:Arial;background:#f2f6fa;padding:40px">

        <div style="max-width:600px;margin:auto;background:white;padding:40px;border-radius:12px">

        <h2 style="color:#0b2b40;">Harsha Pvt Limited</h2>

        <p>Your login OTP is:</p>

        <h1 style="letter-spacing:8px">{otp}</h1>

        <p>This OTP is valid for <b>5 minutes</b>.</p>

        </div>

        </body>
        </html>
        """

        email_message = EmailMultiAlternatives(
            subject,
            text_content,
            settings.EMAIL_HOST_USER,
            [email]
        )

        email_message.attach_alternative(html_content, "text/html")
        email_message.send()

        return redirect("verify")

    return render(request, "login.html")


# -----------------------------
# VERIFY OTP
# -----------------------------
def verify_view(request):

    email = request.session.get("email")

    if request.method == "POST":

        user_otp = request.POST.get("otp")
        real_otp = request.session.get("otp")

        if user_otp == real_otp:
            return redirect("upload")

        else:
            return render(request, "verify.html", {
                "error": "Invalid OTP. Please try again.",
                "email": email
            })

    return render(request, "verify.html", {"email": email})


# -----------------------------
# UPLOAD PAGE
# -----------------------------
def upload_page(request):
    return render(request, "upload.html")


# -----------------------------
# MRI PREDICTION
# -----------------------------
def predict_mri(request):

    if request.method == "POST":

        if "mri" not in request.FILES:
            return render(request, "upload.html", {
                "error": "No MRI image uploaded"
            })

        file = request.FILES["mri"]

        file_path = os.path.join(settings.MEDIA_ROOT, file.name)

        with open(file_path, "wb+") as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        # Image preprocessing
        img = Image.open(file_path).convert("RGB")
        img = img.resize((224, 224))

        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Load model
        model = get_model()

        # Prediction
        prediction = model.predict(img)

        score = float(prediction[0][0])

        if score >= 0.5:
            result = "Alzheimer Detected"
            confidence = round(score * 100, 2)
        else:
            result = "Healthy Brain"
            confidence = round((1 - score) * 100, 2)

        image_url = settings.MEDIA_URL + file.name

        return render(request, "upload.html", {
            "result": result,
            "confidence": confidence,
            "image_url": image_url
        })

    return render(request, "upload.html")