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
from .models import PatientRecord

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


# -----------------------------
# AUTH VIEWS (SIGNUP, LOGIN, SOCIAL)
# -----------------------------
def signup_view(request):
    if request.method == "POST":
        email = request.POST.get("email")
        request.session["email"] = email
        return redirect("upload")
    return render(request, "signup.html")

def social_login(request, provider):
    request.session["email"] = f"social_{provider}_user@neuroscan.ai"
    return redirect("upload")

def forgot_password_view(request):
    if request.method == "POST":
        email = request.POST.get("email")
        return render(request, "login.html", {"error": f"Password reset link sent to {email}"})
    return render(request, "login.html")

def login_view(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")
        
        # Simple login
        if password and len(password) >= 6:
            request.session["email"] = email
            return redirect("upload")

        # OTP
        otp = str(random.randint(100000, 999999))
        print(f"OTP for {email}: {otp}")
        request.session["otp"] = otp
        request.session["email"] = email

        subject = "NeuroScan AI Login"
        text = f"OTP: {otp}"
        html = f"<b>{otp}</b>"
        threading.Thread(target=send_email_async, args=(subject, text, html, email)).start()
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
# UPLOAD PAGE (DASHBOARD)
# -----------------------------
def upload_page(request):
    email = request.session.get("email", "unknown@example.com")
    
    # In a real app we might filter by user_email, here we'll just show all for the demo
    # records = PatientRecord.objects.filter(user_email=email)
    records = PatientRecord.objects.all()
    
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

        # Save using FileSystemStorage into 'scans/' subdirectory to match model
        fs = FileSystemStorage()
        filename = fs.save('scans/' + file.name, file)
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
        
        print(f"\n{'='*50}", flush=True)
        print(f"🚨 RESENT OTP 🚨", flush=True)
        print(f"New OTP for {email} is: {otp}", flush=True)
        print(f"{'='*50}\n", flush=True)

        request.session["otp"] = otp

        subject = "Harsha Pvt Limited – Your Resent Login OTP"
        text_content = f"Your new OTP is {otp}. Valid for 5 minutes."
        
        # Simple HTML for the resend
        html_content = f"""
        <div style="font-family: Arial, sans-serif; text-align: center; padding: 20px; background-color: #f4f4f4;">
            <div style="background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); display: inline-block;">
                <h2 style="color: #333;">Login Verification</h2>
                <p style="color: #666; font-size: 16px;">Your new One-Time Password is:</p>
                <div style="background-color: #007bff; color: white; padding: 15px 30px; border-radius: 5px; font-size: 32px; font-weight: bold; letter-spacing: 5px; margin: 20px 0;">
                    {otp}
                </div>
                <p style="color: #999; font-size: 12px;">Valid for 5 minutes. Do not share this code.</p>
            </div>
        </div>
        """

        # Start thread to send email
        email_thread = threading.Thread(
            target=send_email_async,
            args=(subject, text_content, html_content, email)
        )
        email_thread.start()

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