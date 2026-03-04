import random
import os
import numpy as np
import tensorflow as tf

from PIL import Image

from django.shortcuts import render, redirect
from django.conf import settings
from django.core.mail import EmailMultiAlternatives

# Load your trained model
model = tf.keras.models.load_model("detector/models/alzheimer_binary_model.h5")


# -------------------------------
# LOGIN PAGE (Send OTP)
# -------------------------------

def login_view(request):

    if request.method == "POST":

        email = request.POST.get("email")

        otp = random.randint(100000, 999999)

        request.session['otp'] = otp
        request.session['email'] = email

        subject = "Harsha Pvt Limited"

        text_content = f"""
Welcome to Harsha Pvt Limited

Your OTP is: {otp}
"""

        html_content = f"""
        <html>
        <body style="font-family: Arial; background:#f2f2f2; padding:20px;">

        <div style="max-width:600px;margin:auto;background:white;padding:30px;border-radius:10px;">

        <h2 style="color:#1a73e8;">Welcome to Harsha Pvt Limited</h2>

        <p>Dear User,</p>

        <p>Thank you for accessing our <b>AI Alzheimer MRI Detection Platform</b>.</p>

        <p>Please use the following One-Time Password (OTP) to continue login:</p>

        <div style="text-align:center;margin:30px;">
        <span style="font-size:42px;font-weight:bold;color:#e63946;">
        {otp}
        </span>
        </div>

        <p>This OTP is valid for <b>5 minutes</b>.</p>

        <p style="color:red;">
        Do not share this OTP with anyone.
        </p>

        <p>If you did not request this login, please ignore this email.</p>

        <br>

        <p>
        Best Regards<br>
        <b>Harsha Pvt Limited</b><br>
        AI Healthcare Technology
        </p>

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


# -------------------------------
# VERIFY OTP
# -------------------------------

def verify_view(request):

    if request.method == "POST":

        user_otp = request.POST.get("otp")

        if str(user_otp) == str(request.session.get("otp")):

            return redirect("upload")

        else:

            return render(request, "verify.html", {"error": "Invalid OTP"})

    return render(request, "verify.html")


# -------------------------------
# MRI PREDICTION FUNCTION
# -------------------------------

def predict_mri(image_path):

    img = Image.open(image_path).convert("RGB")
    img = img.resize((224,224))

    img = np.array(img)/255.0
    img = np.expand_dims(img,axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        return "Healthy", pred*100
    else:
        return "Alzheimer Detected", (1-pred)*100


# -------------------------------
# MRI UPLOAD PAGE
# -------------------------------

def upload_mri(request):

    result = None
    confidence = None
    image_url = None

    if request.method == "POST":

        file = request.FILES['mri']

        path = os.path.join("media", file.name)

        with open(path,'wb+') as f:
            for chunk in file.chunks():
                f.write(chunk)

        result, confidence = predict_mri(path)

        image_url = "/" + path

    return render(request,"upload.html",{
        "result":result,
        "confidence":confidence,
        "image_url":image_url
    })