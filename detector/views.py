import os
import random
import threading
import numpy as np
from PIL import Image
import h5py
import urllib.request
import json

from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from django.conf import settings
from django.core.mail import EmailMultiAlternatives
from django.core.files.storage import FileSystemStorage
from .models import PatientRecord, AppUser
import urllib.parse

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

from keras.models import load_model, Sequential
from keras.layers import Dense, Flatten, Dropout, Input, GlobalAveragePooling2D
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import tensorflow as tf

# -----------------------------
# GRAD-CAM UTILS
# -----------------------------
def get_jet_colormap():
    x = np.linspace(0, 1, 256)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    cmap = np.stack([r, g, b], axis=-1) * 255
    return cmap.astype(np.uint8)

def make_gradcam_heatmap(img_array, full_model, last_conv_layer_name="out_relu"):
    base_model = full_model.layers[0] # MobileNetV2
    try:
        last_conv_layer = base_model.get_layer(last_conv_layer_name)
    except ValueError:
        return None

    inner_model = tf.keras.Model(base_model.inputs, last_conv_layer.output)
    
    classifier_input = tf.keras.Input(shape=inner_model.output.shape[1:])
    x = classifier_input
    for layer in full_model.layers[1:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        last_conv_layer_output = inner_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        class_channel = preds[:, 0]
        
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    jet_colors = get_jet_colormap()
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img * (1 - alpha)
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)

# -----------------------------
# MODEL CONFIG
# -----------------------------
MODEL_PATH = os.path.join(settings.BASE_DIR, "detector", "models", "alzheimer_model.h5")
model = None

# -----------------------------
# FIXED MODEL LOADER
# - Uses ImageNet weights for MobileNetV2 (fixes NaN from uninitialized BatchNorm)
# - Loads dense/classifier weights directly from our H5 using exact known paths
# -----------------------------
def get_model():
    global model
    if model is None:
        try:
            if not os.path.exists(MODEL_PATH):
                print(f"Error: Model file not found at {MODEL_PATH}")
                return None

            # Step 1: Build model with ImageNet pretrained MobileNetV2
            # This ensures BatchNorm layers are properly initialized (no NaN)
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
            base_model.trainable = False  # keep base frozen

            model = Sequential([
                base_model,
                GlobalAveragePooling2D(name="global_average_pooling2d"),
                Dense(128, activation="relu", name="dense"),
                Dropout(0.5, name="dropout"),
                Dense(1, activation="sigmoid", name="dense_1")
            ])

            # Step 2: Load ONLY the dense layer weights from our trained H5
            # We know the exact paths from inspecting the H5 file
            with h5py.File(MODEL_PATH, 'r') as f:
                mw = f['model_weights']

                # Dense layer (128 units)
                dense_kernel = np.array(mw['dense']['sequential']['dense']['kernel'])
                dense_bias   = np.array(mw['dense']['sequential']['dense']['bias'])

                # Output Dense layer (1 unit)
                dense1_kernel = np.array(mw['dense_1']['sequential']['dense_1']['kernel'])
                dense1_bias   = np.array(mw['dense_1']['sequential']['dense_1']['bias'])

            # Set the weights on our model layers
            for layer in model.layers:
                if layer.name == 'dense':
                    layer.set_weights([dense_kernel, dense_bias])
                elif layer.name == 'dense_1':
                    layer.set_weights([dense1_kernel, dense1_bias])

            print("Model loaded successfully: ImageNet MobileNetV2 + trained dense weights.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error loading model: {e}")
            return None
    return model


# -----------------------------
# ASYNC EMAIL
# -----------------------------
def send_email_async(subject, text_content, html_content, email):
    try:
        api_key = os.environ.get("BREVO_API_KEY")
        if not api_key:
            print("BREVO_API_KEY is not set. Go to Render and add it!", flush=True)
            return
            
        url = "https://api.brevo.com/v3/smtp/email"
        
        # The sender email must be exactly the email you verify on Brevo
        sender_email = os.environ.get("SENDER_EMAIL", "harshamukundhaaripaka@gmail.com") 
        
        payload = {
            "sender": {"name": "Harsha AI Platform", "email": sender_email},
            "to": [{"email": email}],
            "subject": subject,
            "htmlContent": html_content
        }
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={
            'accept': 'application/json',
            'api-key': api_key,
            'content-type': 'application/json'
        }, method='POST')
        
        with urllib.request.urlopen(req) as response:
            response_data = response.read()
            print(f"Successfully sent Brevo email to {email}", flush=True)
            
    except Exception as e:
        print(f"Failed to send email via Brevo API: {e}", flush=True)


# ============================================================
# BEAUTIFUL OTP EMAIL BUILDER (matches screenshot design)
# ============================================================
def build_otp_email(otp: str, context: str = "Login") -> str:
    """Returns a richly-styled HTML email matching the Harsha Pvt Limited design."""
    digits = "  ".join(list(otp))  # spaced digits like: 4  4  5  4  1  6
    return f"""
<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background-color:#f0f4f8;font-family:Arial,Helvetica,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f0f4f8;padding:30px 0;">
  <tr><td align="center">
  <table width="580" cellpadding="0" cellspacing="0" style="max-width:580px;width:100%;">

    <!-- HEADER -->
    <tr><td style="background:linear-gradient(135deg,#0a2540 0%,#0a4a7a 50%,#0d5fa0 100%);padding:44px 30px;text-align:center;border-radius:14px 14px 0 0;">
      <div style="font-size:52px;margin-bottom:14px;">&#129504;</div>
      <h1 style="color:#ffffff;margin:0;font-size:26px;font-weight:800;letter-spacing:0.5px;">Harsha Pvt Limited</h1>
      <p style="color:#a8d4f0;margin:8px 0 0;font-size:11px;letter-spacing:3px;font-weight:600;">AI ALZHEIMER MRI DETECTION PLATFORM</p>
    </td></tr>

    <!-- BODY -->
    <tr><td style="background:#ffffff;padding:44px 44px 36px;">
      <p style="color:#8a9bb0;font-size:11px;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;margin:0 0 10px;">Login Verification</p>
      <h2 style="color:#0d1f33;font-size:28px;font-weight:800;margin:0 0 22px;">Your One-Time Password</h2>

      <p style="color:#1a2b3c;font-size:15px;margin:0 0 12px;">Hello, <strong>Valued User</strong> &#128075;</p>
      <p style="color:#4a6070;font-size:14px;line-height:1.7;margin:0 0 30px;">
        We received a <strong>{context}</strong> request for your account on the
        <strong>AI Alzheimer MRI Detection Platform</strong>. Use the OTP below to complete your login.
      </p>

      <!-- OTP BOX -->
      <div style="background:#0d1b2a;border-radius:14px;padding:30px 20px;text-align:center;margin-bottom:22px;">
        <p style="color:#6a8aa8;font-size:11px;letter-spacing:3px;text-transform:uppercase;margin:0 0 18px;font-weight:600;">Your OTP Code</p>
        <div style="font-size:44px;font-weight:800;letter-spacing:14px;color:#ffffff;font-family:'Courier New',Courier,monospace;margin-bottom:16px;">{digits}</div>
        <p style="color:#e74c3c;font-size:13px;font-weight:700;margin:0;">&#9940; Valid for 5 minutes only</p>
      </div>

      <!-- BADGES -->
      <table width="100%" cellpadding="4" cellspacing="0" style="margin-bottom:26px;">
        <tr>
          <td width="33%" style="padding:4px;">
            <div style="background:#0d2137;border-radius:10px;padding:14px 8px;text-align:center;">
              <div style="font-size:22px;">&#128274;</div>
              <div style="color:#a8c8e8;font-size:11px;margin-top:6px;font-weight:700;">Secure Login</div>
            </div>
          </td>
          <td width="33%" style="padding:4px;">
            <div style="background:#1a160a;border-radius:10px;padding:14px 8px;text-align:center;">
              <div style="font-size:22px;">&#9889;</div>
              <div style="color:#f0c040;font-size:11px;margin-top:6px;font-weight:700;">One-Time Use</div>
            </div>
          </td>
          <td width="33%" style="padding:4px;">
            <div style="background:#0d2137;border-radius:10px;padding:14px 8px;text-align:center;">
              <div style="font-size:22px;">&#127973;</div>
              <div style="color:#a8c8e8;font-size:11px;margin-top:6px;font-weight:700;">Healthcare AI</div>
            </div>
          </td>
        </tr>
      </table>

      <!-- WARNING -->
      <div style="background:#2d1e21;border-left:4px solid #e74c3c;border-radius:6px;padding:16px 18px;">
        <p style="color:#e8d0c8;font-size:13px;margin:0;line-height:1.65;">
          &#9888; <strong style="color:#f0a07a;">Did not request this?</strong>
          If you did not attempt to log in, please ignore this email. Your account remains secure.
        </p>
      </div>
    </td></tr>

    <!-- FOOTER -->
    <tr><td style="background:#f8fafc;border-top:1px solid #dde6ee;padding:26px 44px;text-align:center;border-radius:0 0 14px 14px;">
      <p style="color:#1a2b3c;font-size:14px;font-weight:800;margin:0 0 5px;">Harsha Pvt Limited</p>
      <p style="color:#7a8fa0;font-size:12px;margin:0 0 18px;">AI Healthcare Technology &bull; MRI Analysis Platform</p>
      <p style="color:#b0bec8;font-size:11px;margin:0 0 5px;">This is an automated message. Please do not reply to this email.</p>
      <p style="color:#b0bec8;font-size:11px;margin:0;">&copy; 2025 Harsha Pvt Limited. All rights reserved.</p>
    </td></tr>

  </table>
  </td></tr>
</table>
</body></html>
"""


# ============================================================
# GOOGLE OAUTH CONSTANTS
# ============================================================
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


# ============================================================
# AUTH VIEWS
# ============================================================

def signup_view(request):
    if request.method == "POST":
        email = request.POST.get("email", "").strip()
        name = request.POST.get("name", "").strip()
        if not email:
            return render(request, "signup.html", {"error": "Email is required."})

        otp = str(random.randint(100000, 999999))
        print(f"Signup OTP for {email}: {otp}")
        request.session["otp"] = otp
        request.session["email"] = email
        request.session["user_name"] = name
        request.session["is_signup"] = True

        subject = "NeuroScan AI – Verify Your Account"
        html = build_otp_email(otp, context="Account Registration")
        threading.Thread(target=send_email_async, args=(subject, otp, html, email)).start()
        return redirect("verify")
    return render(request, "signup.html")


def google_login(request):
    """Step 1 – Redirect user to Google's OAuth consent/account chooser."""
    client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
    if not client_id:
        return render(request, "login.html", {
            "error": "Google login is not configured yet. Please use email/OTP login."
        })
    redirect_uri = request.build_absolute_uri("/google-callback/")
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account",   # forces account chooser every time
    }
    return redirect(GOOGLE_AUTH_URL + "?" + urllib.parse.urlencode(params))


def google_callback(request):
    """Step 2 – Google redirects back here with ?code=... Exchange for user email, send OTP."""
    error_param = request.GET.get("error")
    code = request.GET.get("code")

    if error_param or not code:
        return render(request, "login.html", {
            "error": f"Google login was cancelled or failed: {error_param or 'no code received'}. Please try again."
        })

    client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
    redirect_uri = request.build_absolute_uri("/google-callback/")

    try:
        # Exchange auth code for tokens
        token_data = urllib.parse.urlencode({
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }).encode("utf-8")

        token_req = urllib.request.Request(
            GOOGLE_TOKEN_URL,
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST"
        )
        with urllib.request.urlopen(token_req) as resp:
            token_info = json.loads(resp.read())

        access_token = token_info.get("access_token")
        if not access_token:
            raise Exception("No access token received from Google")

        # Fetch user profile
        info_req = urllib.request.Request(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"}
        )
        with urllib.request.urlopen(info_req) as resp:
            user_info = json.loads(resp.read())

        email = user_info.get("email", "").strip()
        name = user_info.get("name", "Valued User")
        google_id = user_info.get("id", "")

        if not email:
            raise Exception("Google did not return an email address")

        # Generate OTP, store in session
        otp = str(random.randint(100000, 999999))
        print(f"Google OAuth OTP for {email}: {otp}")
        request.session["otp"] = otp
        request.session["email"] = email
        request.session["user_name"] = name
        request.session["google_id"] = google_id
        request.session["is_google"] = True

        # Send the beautiful OTP email
        subject = "NeuroScan AI – Google Login Verification"
        html = build_otp_email(otp, context="Google Login")
        threading.Thread(target=send_email_async, args=(subject, otp, html, email)).start()
        return redirect("verify")

    except Exception as exc:
        print(f"Google OAuth error: {exc}")
        return render(request, "login.html", {
            "error": f"Google authentication failed. Please try again or use email login."
        })


def forgot_password_view(request):
    if request.method == "POST":
        email = request.POST.get("email", "").strip()
        if not email:
            return render(request, "forgot_password.html", {"error": "Please enter a valid email address."})

        otp = str(random.randint(100000, 999999))
        print(f"Password Reset OTP for {email}: {otp}")
        request.session["otp"] = otp
        request.session["email"] = email
        request.session["is_reset"] = True

        subject = "NeuroScan AI – Password Reset"
        html = build_otp_email(otp, context="Password Reset")
        threading.Thread(target=send_email_async, args=(subject, otp, html, email)).start()
        return redirect("verify")

    return render(request, "forgot_password.html")


def login_view(request):
    if request.method == "POST":
        email = request.POST.get("email", "").strip()
        password = request.POST.get("password", "")

        if not email:
            return render(request, "login.html", {"error": "Please enter your email."})

        # Credentials login (password >= 6 chars)
        if password and len(password) >= 6:
            # Create/update user record
            AppUser.objects.get_or_create(email=email, defaults={"name": email.split("@")[0]})
            request.session["email"] = email
            return redirect("upload")

        # Fallback: OTP login
        otp = str(random.randint(100000, 999999))
        print(f"Login OTP for {email}: {otp}")
        request.session["otp"] = otp
        request.session["email"] = email

        subject = "NeuroScan AI – Your Login OTP"
        html = build_otp_email(otp, context="Login")
        threading.Thread(target=send_email_async, args=(subject, otp, html, email)).start()
        return redirect("verify")

    return render(request, "login.html")


# -----------------------------
# VERIFY OTP
# -----------------------------
def verify_view(request):
    email = request.session.get("email")
    if not email:
        return redirect("login")

    if request.method == "POST":
        user_otp = request.POST.get("otp", "").strip()
        real_otp = request.session.get("otp", "")

        if user_otp != real_otp:
            return render(request, "verify.html", {
                "error": "❌ Invalid OTP. Please check your email and try again.",
                "email": email
            })

        # OTP is correct — determine the flow type
        is_reset  = request.session.pop("is_reset",  False)
        is_signup = request.session.pop("is_signup", False)
        is_google = request.session.pop("is_google", False)
        request.session.pop("is_social", False)

        # Persist / retrieve user in DB
        name  = request.session.get("user_name", email.split("@")[0])
        g_id  = request.session.get("google_id", None)

        if is_google and g_id:
            # For Google users, also save google_id
            user, created = AppUser.objects.get_or_create(
                email=email,
                defaults={"name": name, "google_id": g_id}
            )
            if not created and not user.google_id:
                user.google_id = g_id
                user.save(update_fields=["google_id"])
        elif not is_reset:
            # Regular signup / login / social
            AppUser.objects.get_or_create(email=email, defaults={"name": name})

        # Clear OTP from session (one-time use)
        request.session.pop("otp", None)

        if is_reset:
            return render(request, "login.html", {
                "error": "✅ Identity verified! You may now log in with your credentials."
            })

        # All other flows → dashboard
        return redirect("upload")

    return render(request, "verify.html", {"email": email})


# -----------------------------
# UPLOAD PAGE (DASHBOARD)
# -----------------------------
def upload_page(request):
    email = request.session.get("email", "unknown@example.com")
    
    # In a real app we might filter by user_email, here we'll just show all for the demo
    # records = PatientRecord.objects.filter(user_email=email)
    # Filter records by this user's email so data persists across login/logout
    records = PatientRecord.objects.filter(user_email=email)

    total_patients = records.count()
    alzheimer_cases = records.filter(prediction_result__icontains="Alzheimer").count()
    normal_cases = records.filter(prediction_result__icontains="Healthy").count()

    recent_records = records.order_by('-date')[:5]
    
    return render(request, "upload.html", {
        "total_patients": total_patients,
        "alzheimer_cases": alzheimer_cases,
        "normal_cases": normal_cases,
        "recent_records": recent_records
    })

# -----------------------------
# PERFORMANCE PAGE
# -----------------------------
def performance_page(request):
    return render(request, "performance.html")

# -----------------------------
# ALZHEIMER INFO PAGE
# -----------------------------
def info_page(request):
    return render(request, "info.html")



# -----------------------------
# MRI PREDICTION
# -----------------------------
ALLOWED_IMAGE_TYPES = {'image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'}
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

def predict_mri(request):
    if request.method == "POST":
        # ── 1. Check a file was actually submitted ──
        if "mri" not in request.FILES or request.FILES["mri"].size == 0:
            return render(request, "upload.html", {
                "error": "⚠️ No MRI image uploaded. Please select a file before submitting."
            })

        file = request.FILES["mri"]

        # ── 2. File size check (20 MB max) ──
        if file.size > 20 * 1024 * 1024:
            return render(request, "upload.html", {
                "error": "⚠️ File is too large. Maximum allowed size is 20 MB."
            })

        # ── 3. File type validation (extension + MIME) ──
        import os as _os
        ext = _os.path.splitext(file.name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return render(request, "upload.html", {
                "error": f"⚠️ Invalid file type '{ext}'. Please upload a JPEG, PNG, BMP, or TIFF image."
            })

        # ── 4. Save file ──
        fs = FileSystemStorage()
        filename = fs.save('scans/' + file.name, file)
        file_path = fs.path(filename)

        # ── 5. Verify it's a real image ──
        try:
            from PIL import Image as _PILImage
            with _PILImage.open(file_path) as test_img:
                test_img.verify()
        except Exception:
            fs.delete(filename)
            return render(request, "upload.html", {
                "error": "⚠️ The uploaded file could not be read as an image. Please upload a valid MRI scan."
            })

        try:
            # Image preprocessing for MobileNetV2
            img = Image.open(file_path).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img).astype(np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array) # Scale to [-1, 1] as expected by MobileNetV2

            # Load model
            model = get_model()

            if model is None:
                raise Exception("Model could not be loaded")

            # Prediction
            prediction = model.predict(img_array)
            score = float(prediction[0][0])

            # Logic based on sigmoid output (usually 0=Healthy, 1=Alzheimer or vice versa)
            # Based on common datasets for this model:
            if score >= 0.5:
                result = "Alzheimer Detected"
                confidence = round(score * 100, 2)
                risk_level = "High risk"
            else:
                result = "Healthy Brain"
                confidence = round((1 - score) * 100, 2)
                risk_level = "Low risk"

            image_url = settings.MEDIA_URL + filename
            
            # Generate Grad-CAM Heatmap
            heatmap_url = None
            try:
                heatmap = make_gradcam_heatmap(img_array, model, "out_relu")
                if heatmap is not None:
                    cam_filename = "cam_" + filename
                    cam_path = fs.path(cam_filename)
                    save_gradcam(file_path, heatmap, cam_path)
                    heatmap_url = settings.MEDIA_URL + cam_filename
            except Exception as e:
                print(f"Error generating Grad-CAM: {e}", flush=True)

            # Analyze Questionnaire
            q_score = 0
            q_score += int(request.POST.get("q_memory_loss", 0))
            q_score += int(request.POST.get("q_names", 0))
            q_score += int(request.POST.get("q_time_place", 0))
            q_score += int(request.POST.get("q_conversation", 0))
            
            # Analyze Cognitive Score
            cogn_score = request.POST.get("cognitive_score")
            if cogn_score:
                try:
                    cogn_score = int(cogn_score)
                except ValueError:
                    cogn_score = None
            else:
                cogn_score = None

            # Calculate overall risk
            if score >= 0.5 or q_score >= 8 or (cogn_score is not None and cogn_score < 40):
                risk_level = "High risk"
            elif q_score >= 4 or (cogn_score is not None and cogn_score < 70):
                risk_level = "Moderate risk"
            else:
                risk_level = "Low risk"
            
            # Save Patient Record
            email = request.session.get("email", "unknown@example.com")
            patient_name = request.POST.get("patient_name", "Unknown")
            age = request.POST.get("age")
            if age:
                try:
                    age = int(age)
                except ValueError:
                    age = None
            
            record = PatientRecord.objects.create(
                user_email=email,
                patient_name=patient_name,
                age=age,
                scan_image=filename, # filename already includes 'scans/' prefix now
                prediction_result=result,
                confidence=confidence,
                risk_level=risk_level,
                cognitive_score=cogn_score
            )

            return render(request, "upload.html", {
                "result": result,
                "confidence": confidence,
                "image_url": image_url,
                "heatmap_url": heatmap_url,
                "patient_name": patient_name,
                "age": age,
                "record_id": record.id,
                "q_score": q_score,
                "cogn_score": cogn_score,
                "risk_level": risk_level
            })
            
        except Exception as e:
            return render(request, "upload.html", {
                "error": f"An error occurred during prediction: {str(e)}"
            })
    return render(request, "upload.html")


# -----------------------------
# RESEND OTP
# -----------------------------
def resend_otp(request):
    email = request.session.get("email")
    if email:
        otp = str(random.randint(100000, 999999))
        print(f"Resent OTP for {email}: {otp}")

        request.session["otp"] = otp

        subject = "Harsha Pvt Limited – Your Resent Login OTP"
        text_content = f"Your new OTP is {otp}. Valid for 5 minutes."
        html_content = build_otp_email(otp, context="OTP Resend")

        # Start thread to send email
        threading.Thread(
            target=send_email_async,
            args=(subject, text_content, html_content, email)
        ).start()

    return redirect("verify")


# -----------------------------
# LOGOUT VIEW
# -----------------------------
def logout_view(request):
    request.session.flush()
    return redirect("login")

# -----------------------------
# DOWNLOAD PDF REPORT
# -----------------------------
def download_report(request, record_id):
    record = get_object_or_404(PatientRecord, id=record_id)
    
    # Create a file-like buffer to receive PDF data.
    buffer = io.BytesIO()
    
    # Create the PDF object, using the buffer as its "file."
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Draw things on the PDF
    p.setFont("Helvetica-Bold", 20)
    p.drawString(200, height - 50, "Clinical Assessment Report")
    
    p.setFont("Helvetica", 12)
    p.drawString(50, height - 100, f"Patient Name: {record.patient_name}")
    p.drawString(50, height - 120, f"Age: {record.age if record.age else 'N/A'}")
    p.drawString(50, height - 140, f"Date of Assessment: {record.date.strftime('%B %d, %Y')}")
    
    p.line(50, height - 160, width - 50, height - 160)
    
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, height - 200, "AI Diagnosis Results")
    p.setFont("Helvetica", 12)
    p.drawString(50, height - 230, f"Prediction: {record.prediction_result}")
    p.drawString(50, height - 250, f"Confidence: {record.confidence}%")
    p.drawString(50, height - 270, f"Risk Level: {record.risk_level}")
    
    # If the user has a cognitive score
    if record.cognitive_score is not None:
        p.drawString(50, height - 290, f"Cognitive Test Score: {record.cognitive_score}/100")
        
    p.drawString(50, height - 340, "Disclaimer: This AI prediction is for informational purposes only")
    p.drawString(50, height - 360, "and should not be used as a substitute for professional medical advice.")
    
    # Close the PDF object cleanly
    p.showPage()
    p.save()
    
    buffer.seek(0)
    return HttpResponse(buffer, content_type='application/pdf')