from django.contrib import admin
from .models import NewsPrediction


@admin.register(NewsPrediction)
class NewsPredictionAdmin(admin.ModelAdmin):

    list_display = (
        "id",
        "prediction",
        "confidence",
        "fact_check_status",
        "created_at",
    )

    list_filter = (
        "prediction",
        "fact_check_status",
        "created_at",
    )

    search_fields = ("news_text",)

    readonly_fields = ("created_at",)
