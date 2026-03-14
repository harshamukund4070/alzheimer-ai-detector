from django.db import models


class AppUser(models.Model):
    """Stores registered users (email/password or Google OAuth)."""
    email = models.EmailField(unique=True)
    name = models.CharField(max_length=200, blank=True)
    google_id = models.CharField(max_length=200, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_login = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.email


class PatientRecord(models.Model):
    user_email = models.EmailField()  # To associate with the logged-in user
    patient_name = models.CharField(max_length=100)
    age = models.IntegerField(null=True, blank=True)
    scan_image = models.ImageField(upload_to='scans/')
    prediction_result = models.CharField(max_length=100)
    confidence = models.FloatField(null=True, blank=True)
    date = models.DateTimeField(auto_now_add=True)
    risk_level = models.CharField(max_length=50, blank=True)
    cognitive_score = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return f"{self.patient_name} - {self.prediction_result}"
