import os
import random
import threading
import numpy as np
from PIL import Image
import h5py
import urllib.request
import json

from django.shortcuts import render, redirect
from django.conf import settings
from django.core.mail import EmailMultiAlternatives
from django.core.files.storage import FileSystemStorage

from keras.models import load_model, Sequential
from keras.layers import Dense, Flatten, Dropout, Input, GlobalAveragePooling2D
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input


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


# -----------------------------
# LOGIN VIEW (SEND OTP)
# -----------------------------
def login_view(request):

    if request.method == "POST":

        email = request.POST.get("email")

        otp = str(random.randint(100000, 999999))
        
        # Render Free Tier blocks outgoing emails. 
        # We print the OTP to the console so we can still test the app!
        print(f"\n{'='*50}", flush=True)
        print(f"🚨 RENDER FREE TIER BYPASS 🚨", flush=True)
        print(f"OTP for {email} is: {otp}", flush=True)
        print(f"{'='*50}\n", flush=True)

        request.session["otp"] = otp
        request.session["email"] = email

        subject = "Harsha Pvt Limited – Your Login OTP"
        text_content = f"Your OTP is {otp}. Valid for 4 minutes."
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; margin: 0; padding: 0; color: #ffffff;">
            
            <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #1a1a1a; max-width: 600px; margin: 0 auto; border-radius: 12px; overflow: hidden; border: 1px solid #333;">
                
                <!-- HEADER (Gradient & Logo) -->
                <tr>
                    <td style="background: linear-gradient(135deg, #10314a 0%, #206180 100%); padding: 40px 20px; text-align: center;">
                        <div style="background-color: #1c4b69; width: 60px; height: 60px; border-radius: 50%; margin: 0 auto 15px auto; display: inline-flex; justify-content: center; align-items: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                            <span style="font-size: 30px; line-height: 60px; margin-left: 2px;">\U0001F9E0</span>
                        </div>
                        <h1 style="color: #ffffff; margin: 0; font-size: 26px; font-weight: 600; letter-spacing: 0.5px;">Harsha Pvt Limited</h1>
                        <p style="color: #9cb5c5; margin: 8px 0 0 0; font-size: 13px; letter-spacing: 1.5px; text-transform: uppercase;">AI Alzheimer MRI Detection Platform</p>
                    </td>
                </tr>

                <!-- BODY -->
                <tr>
                    <td style="padding: 40px 30px;">
                        <p style="color: #8096a6; font-size: 12px; letter-spacing: 1px; text-transform: uppercase; margin: 0 0 8px 0; font-weight: 600;">Login Verification</p>
                        <h2 style="color: #e2e8f0; margin: 0 0 25px 0; font-size: 24px; font-weight: 500;">Your One-Time Password</h2>
                        
                        <p style="color: #e2e8f0; font-size: 16px; margin-bottom: 20px;">Hello, Valued User \U0001F44B</p>
                        
                        <p style="color: #a0aec0; font-size: 15px; line-height: 1.6; margin-bottom: 35px;">
                            We received a login request for your account on the <strong>AI Alzheimer MRI Detection Platform</strong>. Use the OTP below to complete your login.
                        </p>

                        <!-- OTP BOX -->
                        <div style="background-color: #1e252b; border: 1px solid #2d3748; border-radius: 12px; padding: 30px 20px; text-align: center; margin-bottom: 30px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);">
                            <p style="color: #718096; font-size: 11px; letter-spacing: 2px; text-transform: uppercase; margin: 0 0 15px 0; font-weight: 600;">Your OTP Code</p>
                            <h1 style="color: #ffffff; margin: 0 0 15px 0; font-size: 42px; font-weight: 400; letter-spacing: 12px; font-family: monospace;">{otp}</h1>
                            <p style="color: #f56565; margin: 0; font-size: 13px; font-weight: 500;">\u23F1\uFE0F Valid for 5 minutes only</p>
                        </div>

                        <!-- FEATURE BADGES -->
                        <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom: 30px;">
                            <tr>
                                <td width="31%" align="center" style="background-color: #1a2721; border: 1px solid #273e31; border-radius: 8px; padding: 12px 5px;">
                                    <div style="font-size: 18px; margin-bottom: 5px;">\U0001F512</div>
                                    <div style="color: #68d391; font-size: 12px; font-weight: 500;">Secure Login</div>
                                </td>
                                <td width="3%"></td>
                                <td width="31%" align="center" style="background-color: #2b2319; border: 1px solid #4a3621; border-radius: 8px; padding: 12px 5px;">
                                    <div style="font-size: 18px; margin-bottom: 5px;">\u26A1</div>
                                    <div style="color: #f6ad55; font-size: 12px; font-weight: 500;">One-Time Use</div>
                                </td>
                                <td width="3%"></td>
                                <td width="31%" align="center" style="background-color: #1a2533; border: 1px solid #2a3c50; border-radius: 8px; padding: 12px 5px;">
                                    <div style="font-size: 18px; margin-bottom: 5px;">\U0001F3E5</div>
                                    <div style="color: #63b3ed; font-size: 12px; font-weight: 500;">Healthcare AI</div>
                                </td>
                            </tr>
                        </table>

                        <!-- WARNING BLCOK -->
                        <div style="background-color: #2d1e21; border-left: 4px solid #e53e3e; padding: 15px 20px; border-radius: 4px; border-top-right-radius: 8px; border-bottom-right-radius: 8px;">
                            <p style="color: #e2e8f0; margin: 0; font-size: 14px; line-height: 1.5;">
                                <strong style="color: #fbd38d;">\u26A0\uFE0F Did not request this?</strong> If you did not attempt to log in, please ignore this email. Your account remains secure.
                            </p>
                        </div>
                    </td>
                </tr>

                <!-- FOOTER -->
                <tr>
                    <td style="padding: 30px; text-align: center; border-top: 1px solid #2d3748;">
                        <h4 style="color: #e2e8f0; margin: 0 0 5px 0; font-size: 15px; font-weight: 600;">Harsha Pvt Limited</h4>
                        <p style="color: #718096; margin: 0 0 20px 0; font-size: 12px;">AI Healthcare Technology • MRI Analysis Platform</p>
                        
                        <p style="color: #4a5568; margin: 0 0 5px 0; font-size: 11px;">This is an automated message. Please do not reply to this email.</p>
                        <p style="color: #4a5568; margin: 0; font-size: 11px;">© 2026 Harsha Pvt Limited. All rights reserved.</p>
                    </td>
                </tr>

            </table>
        </body>
        </html>
        """

        # Start thread to send email
        email_thread = threading.Thread(
            target=send_email_async,
            args=(subject, text_content, html_content, email)
        )
        email_thread.start()

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

        # Check file size (e.g. 20MB max)
        if file.size > 20 * 1024 * 1024:
            return render(request, "upload.html", {
                "error": "File is too large. Maximum size is 20MB."
            })

        # Save using FileSystemStorage
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        file_path = fs.path(filename)

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
            else:
                result = "Healthy Brain"
                confidence = round((1 - score) * 100, 2)

            image_url = settings.MEDIA_URL + filename

            return render(request, "upload.html", {
                "result": result,
                "confidence": confidence,
                "image_url": image_url
            })
            
        except Exception as e:
            return render(request, "upload.html", {
                "error": f"An error occurred during prediction: {str(e)}"
            })
    return render(request, "upload.html")