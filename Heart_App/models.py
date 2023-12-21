from django.db import models

# Create your models here.

class AdminDetails(models.Model):
	username = models.CharField(max_length=100,default=None)
	password = models.CharField(max_length=100,default=None)
	class Meta:
		db_table = 'AdminDetails'

class userDetails(models.Model):
	Username 	= models.CharField(max_length=100,default=None,null=True)
	Password 	= models.CharField(max_length=100,default=None,null=True)
	Name 		= models.CharField(max_length=100,default=None,null=True)
	Age 		= models.CharField(max_length=200,default=None,null=True)
	Phone 		= models.CharField(max_length=100,default=None,null=True)
	Email 		= models.CharField(max_length=100,default=None,null=True)
	Address 		= models.CharField(max_length=100,default=None,null=True)
	class Meta:
		db_table = 'userDetails'

class Checkup(models.Model):
	Date= models.CharField(max_length=100,default=None,null=True)
	Oraganised= models.CharField(max_length=100,default=None,null=True)
	Place= models.CharField(max_length=100,default=None,null=True)
	Time= models.CharField(max_length=100,default=None,null=True)

	class Meta:
		db_table = 'Checkup'
