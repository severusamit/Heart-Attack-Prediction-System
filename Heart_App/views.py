from django.shortcuts import render,redirect
from .models import*
from django.contrib import messages
from django.contrib.sessions.models import Session
from django.db import connection
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from django.views import View
import numpy as np
import webbrowser as wb
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import tkinter
from tkinter import *
from django.http import JsonResponse
from .gui import *

# Create your views here.


def Home(request):
	return render(request,"Home.html",{})

def Admin_Login(request):
	if request.method == "POST":
		A_username = request.POST['aname']
		A_password = request.POST['apass']
		if AdminDetails.objects.filter(username = A_username,password = A_password).exists():
			ad = AdminDetails.objects.get(username=A_username, password=A_password)
			print('d')
			messages.info(request,'Admin login is Sucessfull')
			request.session['type_id'] = 'Admin'
			request.session['UserType'] = 'Admin'
			request.session['login'] = "Yes"
			return redirect("/")
		else:
			print('y')
			messages.error(request, 'Error wrong username/password')
			return render(request, "Admin_Login.html", {})
	else:
		return render(request, "Admin_Login.html", {})


def User_Login(request):
	if request.method == "POST":
		C_name = request.POST['aname']
		C_password = request.POST['apass']
		if userDetails.objects.filter(Username=C_name,Password=C_password).exists():
			users = userDetails.objects.all().filter(Username=C_name,Password=C_password)
			messages.info(request,C_name +' logged in')
			request.session['UserId'] = users[0].id
			request.session['type_id'] = 'User'
			request.session['UserType'] = C_name
			request.session['login'] = "Yes"
			return redirect('/')
		else:
			messages.info(request, 'Please Register')
			return redirect("/User_Registeration")
	else:
		return render(request,'User_Login.html',{})

def User_Registeration(request):
	if request.method == "POST":
		Name= request.POST['name']
		Age= request.POST['age']
		Phone= request.POST['phone']
		Email= request.POST['email']
		Address= request.POST['address']
		Username= request.POST['Username']
		Password= request.POST['Password']
		if userDetails.objects.all().filter(Username=Username).exists():
			messages.info(request,"Username Taken")
			return redirect('/User_Registeration')
		else:
			obj = userDetails(
			Name=Name
			,Age=Age
			,Phone=Phone
			,Email=Email
			,Address=Address
			,Username=Username
			,Password=Password)
			obj.save()
			messages.info(request,Name+" Registered")
			return redirect('/User_Login')
	else:
		return render(request,"User_Registeration.html",{})

def Manage_Checkups(request):
	details = Checkup.objects.all()
	return render(request,"Manage_Checkups.html",{'details':details})

def View_User(request):
	details = userDetails.objects.all()
	return render(request,"View_User.html",{'details':details})

def Prediction(request):
	if request.method == "POST":
		age= int(request.POST['age'])
		sex= int(request.POST['Sex'])
		chest_pain_type= int(request.POST['ChestPain'])
		resting_bp= int(request.POST['restingBP'])
		cholesterol= int(request.POST['Cholestrol'])
		fasting_bs= int(request.POST['fasting_bs'])
		resting_ecg= int(request.POST['RestingEcg'])
		max_hr= int(request.POST['MaxHR'])
		exercise_angina= int(request.POST['exercise_angina'])
		oldpeak= float(request.POST['oldpeak'])
		if oldpeak==0:
			oldpeak = 0.0
		st_slope= int(request.POST['slope'])
		print("age :"+str(age))
		print("sex :"+str(sex))
		print("chest_pain_type :"+str(chest_pain_type))
		print("resting_bp :"+str(resting_bp))
		print("cholesterol :"+str(cholesterol))
		print("fasting_bs :"+str(fasting_bs))
		print("resting_ecg :" +str(resting_ecg))
		print("max_hr :"+str(max_hr))
		print("exercise_angina :"+str(exercise_angina))
		print("oldpeak :"+str(oldpeak))
		print("st_slope :"+str(st_slope))
		df = pd.read_csv('heart.csv')
		# Encode the categorical features into numerical values
		encoder = LabelEncoder()
		df['Sex'] = encoder.fit_transform(df['Sex'])
		df['ChestPainType'] = encoder.fit_transform(df['ChestPainType'])
		df['FastingBS'] = encoder.fit_transform(df['FastingBS'])
		df['RestingECG'] = encoder.fit_transform(df['RestingECG'])
		df['ExerciseAngina'] = encoder.fit_transform(df['ExerciseAngina'])
		df['ST_Slope'] = encoder.fit_transform(df['ST_Slope'])
		# Split the dataset into training and testing sets
		X = df.drop('HeartDisease', axis=1)
		y = df['HeartDisease']
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		# Scale the numerical features
		scaler = StandardScaler()
		X_train[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.fit_transform(X_train[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']])
		X_test[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.transform(X_test[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']])
		# Train the logistic regression model
		model = LogisticRegression(random_state=42)
		model.fit(X_train, y_train)
		# Evaluate the performance of the model on the test set
		from sklearn.metrics import precision_score
		from sklearn.metrics import recall_score
		from sklearn.metrics import f1_score
		y_pred = model.predict(X_test)
		accuracy = model.score(X_test, y_test)
		precision = precision_score(y_test, y_pred)
		recall = recall_score(y_test, y_pred)
		f1 = f1_score(y_test, y_pred)
		print('Accuracy:', accuracy)
		print('Precision:', precision)
		print('Recall:', recall)
		print('F1-score:', f1)
		import pickle

		# Save the trained model to a file
		with open('Heartmodel.pkl', 'wb') as file:
		    pickle.dump(model, file)
		# Load the saved model from a file
		with open('Heartmodel.pkl', 'rb') as file:
		    model = pickle.load(file)
		user_input = np.array([[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])
		print(user_input)
		# Scale the numerical features
		user_input[:, [0, 3, 4, 7, 9]] = scaler.transform(user_input[:, [0, 3, 4, 7, 9]])
		# Make a prediction using the loaded model
		prediction = model.predict(user_input)
		if prediction[0] == 0:
		    print("The user does not have heart disease.")
		    messages.info(request,"The user does not have heart disease.")
		else:
		    print("The user has heart disease.")
		    messages.info(request,"The user has heart disease.")
		return render(request,"Prediction.html",{})
	else:
		return render(request,"Prediction.html",{})

def Add_Checkups(request):
	if request.method == "POST":
		Date = request.POST['date']
		Organised = request.POST['Organised']
		Place = request.POST['Address']
		Time = request.POST['time']
		obj = Checkup(Date=Date
					,Oraganised=Organised
					,Place=Place
					,Time=Time)
		obj.save()
		messages.info(request,"Check Up Added")
		return redirect('/Manage_Checkups')
	else:
		return render(request,"Add_Checkups.html",{})

def View_Checkups(request):
	details = Checkup.objects.all()
	return render(request,"View_Checkups.html",{'details':details})

class Message(View):

	def post(self, request):
		msg = request.POST.get('text')
		response = chatbot_response(msg)
		print(response)
		valid=validators.url(response)
		if valid==True:
			data1 = 'True'
			data = {
			'respond': response,'respond1':data1
			}
			return JsonResponse(data)
		else:
			data1 = 'False'
			data = {
			'respond': response,'respond1':data1
			}
			return JsonResponse(data)
		

		#return HttpResponse('data')


	def clean_up_sentence(sentence):
		sentence_words = nltk.word_tokenize(sentence)
		sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
		return sentence_words

	def bow(sentence, words, show_details=True):
		sentence_words = clean_up_sentence(sentence)
		bag = [0] * len(words)
		for s in sentence_words:
			for i, w in enumerate(words):
				if w == s:
					bag[i] = 1
					if show_details:
						print("found in bag: %s" % w)
		return (np.array(bag))


	def predict_class(sentence, model):
		p = bow(sentence, words, show_details=False)
		print(p)
		res = model.predict(np.array([p]))[0]
		print(res)
		ERROR_THRESHOLD = 0.25
		results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
		results.sort(key=lambda x: x[1], reverse=True)
		return_list = []
		for r in results:
			return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
		return return_list

	def getResponse(ints, intents_json):
		tag = ints[0]['intent']
		list_of_intents = intents_json['intents']
		for i in list_of_intents:
			if (i['tag'] == tag):
				result = random.choice(i['responses'])
				print(result)
				break
		return result

	def chatbot_response(msg):
		ints = predict_class(msg, model)
		res = getResponse(ints, intents)
		print(res)
		return res

def ChatWindow(request):
	return render(request,'ChatWindow.html',{})

def Logout(request):
	Session.objects.all().delete()
	return redirect("/")