from django.db import models

class PatientRecord(models.Model):
    user_email = models.EmailField() # To associate with the user who uploaded
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
