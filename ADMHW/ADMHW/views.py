from django.http import HttpResponse
from django.shortcuts import render
from django.views.generic import TemplateView
from .Id3 import predictId3,root
from .NaiveBayes import bayes


def id3Predict (pre_list):
    return predictId3(pre_list, root)

def BayesPredict (pre_list):   
        bayes.process_input(pre_list)
        pos, nega = bayes.get_labels_probabilities(pre_list)
        return pos , nega

def home (request):
    
    if request.method == 'POST':
        try:
            Age = int( request.POST['Age'] )
            Max_Heart_Rate = int ( request.POST['Max_Heart_Rate'])
            Rest_Blood_Pressure = int  (request.POST['Rest_Blood_Pressure'])
    
            Chest_Pain_Type =  request.POST['Chest_Pain_Type']
            Rest_Electro =  request.POST['Rest_Electro']
            Blood_Sugar =  request.POST['Blood_Sugar']
            Exercice_Angina =  request.POST['Exercice_Angina']

            pre_list = [Age , Chest_Pain_Type , Rest_Blood_Pressure , Blood_Sugar , Rest_Electro , Max_Heart_Rate , Exercice_Angina ]
            pos , neg = BayesPredict( [Age , Chest_Pain_Type , Rest_Blood_Pressure , Blood_Sugar , Rest_Electro , Max_Heart_Rate , Exercice_Angina ])
            return render(request,'smartdoctor.html' , {'decision_tree':id3Predict(pre_list) , 'negative_bayes' :  neg   , 'positive_bayes' : pos })

        except:
            
            return render(request,'smartdoctor.html' , 
             {'decision_tree':'Input Field was not in a correct format' ,
             'negative_bayes' : 'Input Field was not in a correct format',
             'positive_bayes' : 'Input Field was not in a correct format' })

    else:
        return render(request,'smartdoctor.html')
 
