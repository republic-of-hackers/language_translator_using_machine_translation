from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    # path('etg/', include(router.urls)),
    path('etg/', views.etg, name='etg'),
    path('gte/', views.gte, name='gte'),
]
