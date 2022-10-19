from django.db import models


class Registration(models.Model):
    attendee_name = models.CharField(max_length=64)
    attendee_id = models.CharField(max_length=40)
    registration_device = models.CharField(max_length=64)
    department = models.CharField(max_length=64)
    image_base64 = models.TextField()
    # face_embedding = ArrayField(ArrayField(models.FloatField()))
    face_embedding = models.JSONField()
    created_on = models.DateTimeField(auto_now_add=True)
    

class FaceVerification(models.Model):
    image_base64 = models.TextField()
    device = models.CharField(max_length=64)


class Attendance(models.Model):
    attendee_name = models.CharField(max_length=64)
    attendee_id = models.CharField(max_length=40)
    device = models.CharField(max_length=64)
    date = models.DateField(auto_now_add=True)
    in_time = models.DateTimeField(blank=True)
    out_time = models.DateTimeField(blank=True, null=True)