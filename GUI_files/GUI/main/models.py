from django.db import models

"""
THESE ARE YOUR DATABASES BRO
"""


# Create your models here.
class Option(models.Model):
    name = models.CharField(max_length=200)
    summary = models.CharField(max_length=200)
    slug = models.CharField(max_length=200)

    def __str__(self):
        return self.name


# class PatientEntry(models.Model):
#     pid = models.CharField(max_length=5)
#     slug = models.CharField(max_length=50)
#
#     class Meta:
#         # Gives the proper plural name for admin
#         verbose_name_plural = "Patient Entries"
#
#
#     def __str__(self):
#         return f"Patient-{self.pid}"


class PatientResult(models.Model):
    # Temporary, this doesn't make sense later on . . .
    pid = models.CharField(verbose_name="Patient ID", max_length=5)
    predicted_age = models.CharField(verbose_name="Predicted Age", max_length=50)
    predicted_sex = models.CharField(verbose_name="Predicted Sex", max_length=50)
    predicted_premature = models.CharField(verbose_name="Predicted Premature", max_length=200)
    segmented_address = models.CharField(max_length=200)

    class Meta:
        # Gives the proper plural name for admin
        verbose_name_plural = "Patient Results"


    def __str__(self):
        return f"Patient-{self.pid} records"


class SessionDatabase(models.Model):
    participant_id = models.CharField(verbose_name="participant_id", max_length=100)
    session_id = models.CharField(verbose_name="session_id", max_length=100, unique=True)
    gender = models.CharField(verbose_name="gender", max_length=100)
    birth_age = models.CharField(verbose_name="birth_age", max_length=100)
    birth_weight = models.CharField(verbose_name="birth_weight", max_length=100)
    singleton = models.CharField(verbose_name="singleton", max_length=100)
    scan_age = models.CharField(verbose_name="scan_age", max_length=50)
    scan_number = models.CharField(verbose_name="scan_number", max_length=50)
    radiology_score = models.CharField(verbose_name="radiology_score", max_length=200)
    sedation = models.CharField(verbose_name="sedation", max_length=200)

    class Meta:
        # Gives the proper plural name for admin
        verbose_name_plural = "Session Database"

    def __str__(self):
        return f"Session ID: {self.session_id}"


class GreyMatterVolume(models.Model):
    session_id = models.CharField(verbose_name="session_id", max_length=200, unique=True)
    filename = models.CharField(verbose_name="filename", max_length=200)
    path = models.CharField(verbose_name="path", max_length=200)
    filepath = models.CharField(verbose_name="filepath", max_length=200)

    def __str__(self):
        return f"Session ID: {self.session_id}"

# samples = [sitk.GetArrayFromImage(sitk.ReadImage(f"{data_dir}/greymatter/wc1sub-{ID}_T1w.nii.gz", sitk.sitkFloat32) for ID in self.ids]

# class AvailableSession(models.Model):
#     session_id = models.CharField(verbose_name="session_id", max_length=200)
