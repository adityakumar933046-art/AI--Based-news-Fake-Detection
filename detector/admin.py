from django.contrib import admin

from .models import NewsPrediction


@admin.register(NewsPrediction)
class NewsPredictionAdmin(admin.ModelAdmin):

    list_display = (
        'id',
        'prediction',
        'confidence',
        'created_at'
    )

    list_filter = ('prediction', 'created_at')

    search_fields = ('news_text',)
