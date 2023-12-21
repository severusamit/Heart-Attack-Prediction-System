from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .import views
from Heart_App.views import Message


urlpatterns = [
				
				path('',views.Home,name='Home'),
				path('User_Login/',views.User_Login,name='User_Login'),
				path('Admin_Login/',views.Admin_Login,name='Admin_Login'),
				path('User_Registeration/',views.User_Registeration,name='User_Registeration'),
				path('Manage_Checkups/',views.Manage_Checkups,name='Manage_Checkups'),
				path('View_User/',views.View_User,name='View_User'),
				path('Prediction/',views.Prediction,name='Prediction'),
				path('Add_Checkups/',views.Add_Checkups,name='Add_Checkups'),
				path('View_Checkups/',views.View_Checkups,name='View_Checkups'),
				path('Message/', Message.as_view(),name='Message'),
				path('ChatWindow/',views.ChatWindow,name='ChatWindow'),
				path('Logout/',views.Logout,name='Logout'),
					
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)