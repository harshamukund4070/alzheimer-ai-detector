import os
import random
import numpy as np
from PIL import Image
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.mail import EmailMultiAlternatives
from tensorflow.keras.models import load_model

# Use BASE_DIR to build model path — prevents deployment errors
MODEL_PATH = os.path.join(settings.BASE_DIR, "detector", "models", "alzheimer_model.h5")
model = load_model(MODEL_PATH, compile=False)


# -----------------------------
# LOGIN VIEW (SEND OTP)
# -----------------------------
def login_view(request):

    if request.method == "POST":

        email = request.POST.get("email")

        # generate 6 digit OTP
        otp = str(random.randint(100000, 999999))

        # store in session
        request.session["otp"] = otp
        request.session["email"] = email

        # -----------------------------------------------
        # BEAUTIFUL HTML EMAIL
        # -----------------------------------------------
        subject      = "Harsha Pvt Limited – Your Login OTP"
        text_content = f"Your OTP is {otp}. Valid for 5 minutes."
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Login OTP</title>
</head>
<body style="margin:0;padding:0;background:#eef2f7;font-family:'Segoe UI',Arial,sans-serif;">

  <table width="100%" cellpadding="0" cellspacing="0" style="background:#eef2f7;padding:40px 0;">
    <tr>
      <td align="center">

        <table width="600" cellpadding="0" cellspacing="0"
               style="background:#ffffff;border-radius:20px;overflow:hidden;
                      box-shadow:0 8px 32px rgba(0,40,100,0.12);max-width:600px;width:100%;">

          <!-- HEADER -->
          <tr>
            <td style="background:linear-gradient(135deg,#0b2b40 0%,#1d4e6b 60%,#2f7fb5 100%);
                       padding:48px 40px 36px;text-align:center;">
              <div style="display:inline-block;background:rgba(255,255,255,0.15);
                          border-radius:50%;width:72px;height:72px;line-height:72px;
                          font-size:36px;margin-bottom:18px;">
                &#129504;
              </div>
              <h1 style="margin:0;color:#ffffff;font-size:26px;font-weight:700;
                         letter-spacing:0.5px;line-height:1.3;">
                Harsha Pvt Limited
              </h1>
              <p style="margin:6px 0 0;color:rgba(255,255,255,0.75);
                        font-size:13px;letter-spacing:1.5px;text-transform:uppercase;">
                AI Alzheimer MRI Detection Platform
              </p>
            </td>
          </tr>

          <!-- BODY -->
          <tr>
            <td style="padding:44px 48px 36px;">

              <p style="margin:0 0 8px;color:#5f7d9c;font-size:13px;
                        text-transform:uppercase;letter-spacing:1.2px;font-weight:600;">
                Login Verification
              </p>
              <h2 style="margin:0 0 20px;color:#0a1929;font-size:22px;font-weight:700;">
                Your One-Time Password
              </h2>

              <p style="margin:0 0 28px;color:#4a5568;font-size:15px;line-height:1.7;">
                Hello, <b>Valued User</b> &#128075;<br><br>
                We received a login request for your account on the
                <b>AI Alzheimer MRI Detection Platform</b>.
                Use the OTP below to complete your login.
              </p>

              <!-- OTP BOX -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:28px;">
                <tr>
                  <td align="center"
                      style="background:linear-gradient(135deg,#f0f7ff,#e6f0f9);
                             border:2px solid #bdd8f0;border-radius:16px;padding:32px 20px;">
                    <p style="margin:0 0 10px;color:#5f7d9c;font-size:12px;
                               text-transform:uppercase;letter-spacing:2px;font-weight:600;">
                      Your OTP Code
                    </p>
                    <span style="font-size:52px;font-weight:800;letter-spacing:18px;
                                 color:#0b2b40;font-family:'Courier New',monospace;
                                 padding-left:18px;">
                      {otp}
                    </span>
                    <p style="margin:16px 0 0;color:#e53e3e;font-size:13px;font-weight:600;">
                      &#9201; Valid for <b>5 minutes</b> only
                    </p>
                  </td>
                </tr>
              </table>

              <!-- INFO PILLS -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:28px;">
                <tr>
                  <td width="33%" align="center"
                      style="background:#f0fdf4;border-radius:12px;padding:14px 8px;">
                    <div style="font-size:20px;margin-bottom:4px;">&#128274;</div>
                    <div style="color:#276749;font-size:12px;font-weight:600;">Secure Login</div>
                  </td>
                  <td width="4%"></td>
                  <td width="33%" align="center"
                      style="background:#fff7ed;border-radius:12px;padding:14px 8px;">
                    <div style="font-size:20px;margin-bottom:4px;">&#9889;</div>
                    <div style="color:#c05621;font-size:12px;font-weight:600;">One-Time Use</div>
                  </td>
                  <td width="4%"></td>
                  <td width="33%" align="center"
                      style="background:#ebf8ff;border-radius:12px;padding:14px 8px;">
                    <div style="font-size:20px;margin-bottom:4px;">&#127973;</div>
                    <div style="color:#2b6cb0;font-size:12px;font-weight:600;">Healthcare AI</div>
                  </td>
                </tr>
              </table>

              <!-- WARNING BOX -->
              <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                  <td style="background:#fff5f5;border-left:4px solid #fc8181;
                             border-radius:0 10px 10px 0;padding:14px 18px;">
                    <p style="margin:0;color:#742a2a;font-size:13px;line-height:1.6;">
                      &#9888;&#65039; <b>Did not request this?</b> If you did not attempt to log in,
                      please ignore this email. Your account remains secure.
                    </p>
                  </td>
                </tr>
              </table>

            </td>
          </tr>

          <!-- DIVIDER -->
          <tr>
            <td style="padding:0 48px;">
              <hr style="border:none;border-top:1px solid #e2ecf4;margin:0;">
            </td>
          </tr>

          <!-- FOOTER -->
          <tr>
            <td style="padding:28px 48px 36px;text-align:center;">
              <p style="margin:0 0 6px;color:#0a1929;font-size:14px;font-weight:700;">
                Harsha Pvt Limited
              </p>
              <p style="margin:0 0 14px;color:#5f7d9c;font-size:12px;">
                AI Healthcare Technology &middot; MRI Analysis Platform
              </p>
              <p style="margin:0;color:#a0aec0;font-size:11px;line-height:1.6;">
                This is an automated message. Please do not reply to this email.<br>
                &copy; 2025 Harsha Pvt Limited. All rights reserved.
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>

</body>
</html>
"""

        # Send as both plain text (fallback) + HTML
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
# UPLOAD PAGE (GET only)
# -----------------------------
def upload_page(request):
    return render(request, "upload.html")


# -----------------------------
# MRI PREDICTION (POST only)
# -----------------------------
def predict_mri(request):

    if request.method == "POST":

        # Check file was actually uploaded
        if "mri" not in request.FILES:
            return render(request, "upload.html", {"error": "No file uploaded"})

        file = request.FILES["mri"]

        # Save uploaded file to MEDIA_ROOT
        file_path = os.path.join(settings.MEDIA_ROOT, file.name)
        with open(file_path, "wb+") as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        # Preprocess image
        img = Image.open(file_path).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Run prediction
        prediction = model.predict(img)

        # 2-CLASS MODEL: single sigmoid output
        # Close to 0 = Healthy Brain, Close to 1 = Alzheimer Detected
        score = float(prediction[0][0])

        if score >= 0.5:
            result = "Alzheimer Detected"
            confidence = round(score * 100, 2)
        else:
            result = "Healthy Brain"
            confidence = round((1 - score) * 100, 2)

        # Build image URL for preview in template
        image_url = settings.MEDIA_URL + file.name

        return render(request, "upload.html", {
            "result": result,
            "confidence": confidence,
            "image_url": image_url
        })

    return render(request, "upload.html")