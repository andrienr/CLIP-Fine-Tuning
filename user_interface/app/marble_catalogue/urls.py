from django.urls import path
from . import views


urlpatterns = [
    path('', views.index),
    path('marbles/<str:search_query>',
         views.Marbles.as_view()),
]
