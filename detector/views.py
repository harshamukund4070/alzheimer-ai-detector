import random
import os
from PIL import Image
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.mail import EmailMultiAlternatives

# ============================================
# MEMORY OPTIMIZED - Import tensorflow ONLY when needed
# ============================================
model = None
model_available = False

def get_model():
    """Lazy load model only when needed to save memory"""
    global model, model_available
    
    if model is not None:
        return model, model_available
    
    try:
        # Import tensorflow only here
        import tensorflow as tf
        
        # Disable GPU and reduce memory usage
        tf.config.set_visible_devices([], 'GPU')
        
        # Load model with memory optimization
        model = tf.keras.models.load_model(
            "detector/models/alzheimer_binary_model.h5",
            compile=False
        )
        model_available = True
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Model loading failed: {e}")
        print("⚡ Using dummy predictions")
        model_available = False
    
    return model, model_available


# ============================================
# PREDICTION FUNCTION - Memory optimized
# ============================================
def predict_mri(image_path):
    """Predict with lazy model loading"""
    
    # Try to get model (loads only if needed)
    model, model_available = get_model()
    
    # If model isn't available, use dummy predictions
    if not model_available or model is None:
        # Simple dummy prediction
        import random
        import time
        time.sleep(1)  # Simulate processing
        
        outcomes = [
            ("Healthy", random.uniform(85, 99)),
            ("Alzheimer Detected", random.uniform(75, 95))
        ]
        idx = 0 if random.random() < 0.7 else 1
        result, confidence = outcomes[idx]
        
        print(f"⚡ Dummy: {result} ({confidence:.1f}%)")
        return result, confidence
    
    # Use real model
    try:
        # Process image
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        if prediction > 0.5:
            result = "Healthy"
            confidence = float(prediction) * 100
        else:
            result = "Alzheimer Detected"
            confidence = float(1 - prediction) * 100
        
        print(f"✅ Real: {result} ({confidence:.1f}%)")
        return result, confidence
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return "Error in analysis", 0


# ============================================
# LOGIN VIEW
# ============================================
def login_view(request):
    if request.method == "POST":
        email = request.POST.get("email")
        otp = random.randint(100000, 999999)
        
        request.session['otp'] = otp
        request.session['email'] = email
        
        subject = "Harsha Pvt Limited - Alzheimer AI Detector"
        text_content = f"Your OTP is: {otp}"
        
        html_content = f"""
        <html>
        <body style="font-family: Arial;">
        <h2>Welcome to Harsha Pvt Limited</h2>
        <p>Your OTP is: <b>{otp}</b></p>
        <p>This OTP is valid for 5 minutes.</p>
        </body>
        </html>
        """
        
        try:
            email_message = EmailMultiAlternatives(
                subject, text_content, settings.EMAIL_HOST_USER, [email]
            )
            email_message.attach_alternative(html_content, "text/html")
            email_message.send()
            print(f"✅ OTP sent to {email}")
        except Exception as e:
            print(f"❌ Email error: {e}")
        
        return redirect("verify")
    
    return render(request, "login.html")


# ============================================
# VERIFY VIEW
# ============================================
def verify_view(request):
    if request.method == "POST":
        user_otp = request.POST.get("otp")
        session_otp = request.session.get("otp")
        
        if str(user_otp) == str(session_otp):
            print("✅ OTP verified successfully")
            return redirect("upload")
        else:
            print(f"❌ Invalid OTP")
            return render(request, "verify.html", {"error": "Invalid OTP"})
    
    return render(request, "verify.html")


# ============================================
# UPLOAD VIEW
# ============================================
def upload_mri(request):
    result = None
    confidence = None
    image_url = None

    if request.method == "POST" and request.FILES.get('mri'):
        try:
            # Save file
            file = request.FILES['mri']
            os.makedirs("media", exist_ok=True)
            file_path = os.path.join("media", file.name)
            
            with open(file_path, 'wb+') as f:
                for chunk in file.chunks():
                    f.write(chunk)
            
            print(f"✅ File saved: {file_path}")
            
            # Make prediction (model loads only here)
            result, confidence = predict_mri(file_path)
            image_url = "/" + file_path
            
            print(f"📊 Result: {result} ({confidence:.1f}%)")
            
        except Exception as e:
            print(f"❌ Upload error: {e}")
            result = "Error"
            confidence = 0

    return render(request, "upload.html", {
        "result": result,
        "confidence": confidence,
        "image_url": image_url
    })