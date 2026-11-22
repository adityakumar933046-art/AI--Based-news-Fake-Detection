
from django.db import models


class NewsPrediction(models.Model):

    news_text = models.TextField()

    prediction = models.CharField(max_length=20)

    confidence = models.FloatField()

    # Google Fact Check API results
    fact_check_results = models.JSONField(default=list, blank=True)

    fact_check_status = models.CharField(
        max_length=20,
        default="NOT_CHECKED"
    )

    fact_check_error = models.TextField(blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.prediction
