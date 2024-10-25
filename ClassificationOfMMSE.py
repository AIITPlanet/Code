import numpy as np
import Write_Metric_To_Excel as wte
import matplotlib.pyplot as plt
#Clasifiers
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report,precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, KFold
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import joblib
import datetime
import json

import pandas as pd


CV=11
f_NR=1
PC_NAME="TSERVER_V2_Learning_NoDem"
GivName="Learning_seg_V2"
#PC_NAME="BTHLAPTOP"
#PC_NAME="HOME"


data = pd.read_excel("Dataset/Edent_Snac_Data_20230901.xlsx", usecols='C:AP')
data2 = pd.read_excel("Dataset/Edent_Snac_Data_20230901.xlsx", usecols='AQ')
#All Parameters
# Attributs=['BAS_Kön','BAS_Age','BAS_Education_PWD','BAS_D1_PWD','BAS_Marital_Status_PWD','BAS_Avtagbar_protetik','BAS_Hel_överkäksprotes_2a','BAS_Delprotes_överkäke_2b','BAS_Hel_underkäksprotes_2c',	
#            'BAS_Delprotes_underkäke_2d','BAS_Förekomst_av_Protesstomatit','BAS_Förekomst_av_Lichen','BAS_Förekomst_av_Munvinkelgrader','BAS_Förekomst_av_Decubitus','BAS_Förekomst_av_Leukoplaki','BAS_Candidainfection',
#             'BAS_Spegeltest','BAS_Totalt_antal_tänder_bettet','BAS_Totalantal_4mmfickor_procent','BAS_Totalantal_5mmfickor_procent','BAS_Totalantal_6mmochöverfickor','BAS_Totalt_antal_blödningar_bettet_genom_antal_ytor_procent',	
#                 'BAS_Totalt_antal_plackytor_bettet_genom_tandytor_bettet_procent','BAS_Antal_furkaturer','BAS_Salivsekretion_tuggslim',
#                 'BAS_E214','BAS_E215a','BAS_E215b','BAS_E215c','BAS_E218','BAS_E220','BAS_E221','BAS_E231','BAS_235',
#                 'BAS_E235c','BAS_E256','BAS_E257','BAS_E259','BAS_D19_Röker_du']
#18 parameters
# Attributs=['BAS_Kön','BAS_Age','BAS_Education_PWD','BAS_D1_PWD','BAS_Hel_överkäksprotes_2a','BAS_Delprotes_överkäke_2b','BAS_Hel_underkäksprotes_2c',	
#            'BAS_Delprotes_underkäke_2d','BAS_Spegeltest','BAS_Totalt_antal_tänder_bettet','BAS_Totalantal_4mmfickor_procent','BAS_Totalantal_5mmfickor_procent','BAS_Totalantal_6mmochöverfickor','BAS_Totalt_antal_blödningar_bettet_genom_antal_ytor_procent',	
#                 'BAS_Totalt_antal_plackytor_bettet_genom_tandytor_bettet_procent','BAS_Antal_furkaturer',
#                 'BAS_E221','BAS_235']

Attributs=['BAS_Kön','BAS_Age','BAS_Education_PWD','BAS_Hel_överkäksprotes_2a','BAS_Delprotes_överkäke_2b','BAS_Hel_underkäksprotes_2c',	
           'BAS_Delprotes_underkäke_2d','BAS_Spegeltest','BAS_Totalt_antal_tänder_bettet','BAS_Totalantal_4mmfickor_procent','BAS_Totalantal_5mmfickor_procent','BAS_Totalantal_6mmochöverfickor','BAS_Totalt_antal_blödningar_bettet_genom_antal_ytor_procent',	
                'BAS_Totalt_antal_plackytor_bettet_genom_tandytor_bettet_procent','BAS_E221','BAS_235']

X=[]
Y=[]
print(len(Attributs))
#RF Results
precisionRFVal = []
recallRFVal = []
f1RFVal = []
accuracyRFVal= []
# precisionRFTest = []
# recallRFTest = []
# f1RFTest = []
# accuracyRFTest= []

#SVM Results 
precisionSVMVal = []
recallSVMVal = []
f1SVMVal = []
accuracySVMVal= []
# precisionSVMTest = []
# recallSVMTest = []
# f1SVMTest = []
# accuracySVMTest= []

#CB Results
precisionCBVal = []
recallCBVal = []
f1CBVal = []
accuracyCBVal= []
# precisionCBTest = []
# recallCBTest = []
# f1CBTest = []
# accuracyCBTest= []

kFoldCV=[]
pathEx="Reports/Results"
demen=[]

for j in range(len(data)):
    row=[]
    for x in range(len(Attributs)):
        row.append(data[Attributs[x]][j])
    X.append(row)
    Y.append(data2['Label'][j])
    if(data['BAS_D1_PWD'][j]==1):
        if(data2['Label'][j]==0):
            demen.append(1)
    #print(row)
    #print(str(len(row)))
X=np.array(X)
Y=np.array(Y)

print("Data:")
print(str(len(X)))
print(str(len(Y)))

print("Dem:")
print(str(len(demen)))



# Split data into training and testing sets
X_train_Main, X_test_Main, y_train_Main, y_test_Main = train_test_split(X, Y, test_size=0.2, random_state=42)
print("Training dataset splited to two sections")
#--------Number of parameters
print("Spliting Training Data set Completed")
print("Data :",str(len(X_train_Main)))
print("Labels :",str(len(y_train_Main)))
print("Spliting Validation Data set Completed")
print("Data :",str(len(X_test_Main)))
print("Labels :",str(len(y_test_Main)))

for i in range(2,CV):
    for j in range(2,CV):
        print("Nested_outer_inner_"+str(i)+" X "+str(j))

        # Create all classifiers
        RF_Classifier = RandomForestClassifier()
        SVM_Classifier= SVC()
        CB_Classifier=CatBoostClassifier(verbose=False)

        # Set up parameter grid for hyperparameter tuning to rondom forest clasifier
        param_grid_RF = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        # Define the parameter grid for hyperparameter tuning to SVM
        param_grid_SVM = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'degree': [2, 3, 4],
        }

        # Define the parameter grid for hyperparameter tuning to CB
        param_grid_CB = {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5]
        }
        countloop=0
        outer_cv = KFold(n_splits=i, shuffle=True, random_state=42)
        # Lists to store RF the results
        outer_scores_RF= []
        # Lists to store SVM the results
        outer_scores_SVM = []
        # Lists to store CB the results
        outer_scores_CB = []

        for train_idx, test_idx in outer_cv.split(X_train_Main):
            X_train, X_test = X_train_Main[train_idx], X_train_Main[test_idx]
            y_train, y_test = y_train_Main[train_idx], y_train_Main[test_idx]
            countloop +=1
            # Inner cross-validation loop for hyperparameter tuning
            inner_cv = KFold(n_splits=j, shuffle=True, random_state=42)

            # Grid search for hyperparameter tuning
            grid_search_RF = GridSearchCV(estimator=RF_Classifier, param_grid=param_grid_RF, cv=inner_cv)
            grid_search_SVM = GridSearchCV(estimator=SVM_Classifier, param_grid=param_grid_SVM, cv=inner_cv)
            grid_search_CB = GridSearchCV(estimator=CB_Classifier, param_grid=param_grid_CB, cv=inner_cv)
            grid_search_RF.fit(X_train, y_train)
            grid_search_SVM.fit(X_train, y_train)
            grid_search_CB.fit(X_train, y_train)

            # Get the best model from the grid search
            best_rf_model = grid_search_RF.best_estimator_
            best_svm_model = grid_search_SVM.best_estimator_
            best_cb_model = grid_search_CB.best_estimator_

            train_sizes, train_scores, test_scores = learning_curve(best_rf_model, X_train, y_train, cv=CV,n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
            plt.figure()
            plt.title("Learning Curve for Random Forest Classifier")
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            plt.grid()

            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
            plt.legend(loc="best")
            plt.savefig(f"Reports/LearningCurves/Learning_Curve_RF_{GivName}_{i}X{j}_seg_{countloop}.png")
            plt.close()

            train_sizes, train_scores, test_scores = learning_curve(best_svm_model, X_train, y_train, cv=CV,n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
            plt.figure()
            plt.title("Learning Curve for Support Vector Machine Classifier")
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            plt.grid()

            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
            plt.legend(loc="best")
            plt.savefig(f"Reports/LearningCurves/Learning_Curve_SVM_{GivName}_{i}X{j}_{countloop}.png")
            plt.close()

            train_sizes, train_scores, test_scores = learning_curve(best_cb_model, X_train, y_train, cv=CV,n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
            plt.figure()
            plt.title("Learning Curve for CatBoost Classifier")
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            plt.grid()

            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
            plt.legend(loc="best")
            plt.savefig(f"Reports/LearningCurves/Learning_Curve_CB_{GivName}_{i}X{j}_{countloop}.png")
            plt.close()
            # Get the best Parameters
            best_rf_params = grid_search_RF.best_params_
            best_svm_params = grid_search_SVM.best_params_
            best_cb_params = grid_search_CB.best_params_

            # Evaluate the best model on the outer test fold
            outer_score_RF = best_rf_model.score(X_test, y_test)
            outer_scores_RF.append(outer_score_RF)
            outer_score_SVM = best_svm_model.score(X_test, y_test)
            outer_scores_SVM.append(outer_score_SVM)
            outer_score_CB = best_cb_model.score(X_test, y_test)
            outer_scores_CB.append(outer_score_CB)
            print("Progress : "+str(i))
      

        # Save the trained model to disk
        joblib.dump(grid_search_RF, './TrainedModels/Analyse_RF_Neasted_Se_'+str(i)+'X'+str(j)+'_'+PC_NAME+'.pkl')
        joblib.dump(grid_search_SVM, './TrainedModels/Analyse_SVM_Neasted_Se_'+str(i)+'X'+str(j)+'_'+PC_NAME+'.pkl')
        joblib.dump(grid_search_CB, './TrainedModels/Analyse_CB_Neasted_Se_'+str(i)+'X'+str(j)+'_'+PC_NAME+'.pkl')

         # Save the best hyperparameters to a JSON file
        with open('Reports/BestParameters/RF_Neasted_'+str(i)+'X'+str(j)+'_hyperparameters.json', 'w') as json_file:
            json.dump(best_rf_params, json_file)
        with open('Reports/BestParameters/SVM_Neasted_'+str(i)+'X'+str(j)+'_hyperparameters.json', 'w') as json_file:
            json.dump(best_svm_params, json_file)
        with open('Reports/BestParameters/CD_Neasted_'+str(i)+'X'+str(j)+'_hyperparameters.json', 'w') as json_file:
            json.dump(best_cb_params, json_file)
        
        kFoldCV.append(str(i)+'X'+str(j))

        # Calculate and print the classification report for RF
        predicted_classes_RF = grid_search_RF.predict(X_test_Main)
        report_RF = classification_report(y_test_Main, predicted_classes_RF)
        #print("Classification Report For RF:")
        #print(report_RF)

        # Calculate precision, recall, F1 score, and accuracy
        precisionRFVal.append(precision_score(y_test_Main, predicted_classes_RF, average='macro'))
        recallRFVal.append(recall_score(y_test_Main, predicted_classes_RF, average='macro'))
        f1RFVal.append(f1_score(y_test_Main, predicted_classes_RF, average='macro'))
        accuracyRFVal.append(accuracy_score(y_test_Main, predicted_classes_RF))

        # Calculate and print the classification report for SVM
        predicted_classes_SVM = grid_search_SVM.predict(X_test_Main)
        report_SVM = classification_report(y_test_Main, predicted_classes_SVM)
        # print("Classification Report For SVM:")
        # print(report_SVM)

        # Calculate precision, recall, F1 score, and accuracy
        precisionSVMVal.append(precision_score(y_test_Main, predicted_classes_SVM, average='macro'))
        recallSVMVal.append(recall_score(y_test_Main, predicted_classes_SVM, average='macro'))
        f1SVMVal.append(f1_score(y_test_Main, predicted_classes_SVM, average='macro'))
        accuracySVMVal.append(accuracy_score(y_test_Main, predicted_classes_SVM))

        # Calculate and print the classification report for SCB
        predicted_classes_CB = grid_search_CB.predict(X_test_Main)
        report_CB = classification_report(y_test_Main, predicted_classes_CB)
        # print("Classification Report For CB:")
        # print(report_CB)

        # Calculate precision, recall, F1 score, and accuracy
        precisionCBVal.append(precision_score(y_test_Main, predicted_classes_CB, average='macro'))
        recallCBVal.append(recall_score(y_test_Main, predicted_classes_CB, average='macro'))
        f1CBVal.append(f1_score(y_test_Main, predicted_classes_CB, average='macro'))
        accuracyCBVal.append(accuracy_score(y_test_Main, predicted_classes_CB))


      
                #print(str(x.strftime("%Y")+x.strftime("%m")+x.strftime("%d")+x.strftime("%H")+"_data.txt"))
        outfile=open("Reports/Nested_Reports/Validation_Se"+str(f_NR)+"_"+PC_NAME+"_data.txt", "a")
        outfile.write("Nested : "+str(i)+"X"+str(j)+" : \nRF :\n"+report_RF+"\nSVM :\n"+report_SVM+"\nCB :\n"+report_CB)
        outfile.write('\n')
        outfile.close()
# Write all results to Excel        
wte.WriteToExcel(pathEx,"RFVal"+PC_NAME,kFoldCV,precisionRFVal,recallRFVal,f1RFVal,accuracyRFVal)
#wte.WriteToExcel(pathEx,"RFTest"+PC_NAME,kFoldCV,precisionRFTest,recallRFTest,f1RFTest,accuracyRFTest)

wte.WriteToExcel(pathEx,"SVMVal"+PC_NAME,kFoldCV,precisionSVMVal,recallSVMVal,f1SVMVal,accuracySVMVal)
#wte.WriteToExcel(pathEx,"SVMTest"+PC_NAME,kFoldCV,precisionSVMTest,recallSVMTest,f1SVMTest,accuracySVMTest)

wte.WriteToExcel(pathEx,"CBVal"+PC_NAME,kFoldCV,precisionCBVal,recallCBVal,f1CBVal,accuracyCBVal)
#wte.WriteToExcel(pathEx,"CBTest"+PC_NAME,kFoldCV,precisionRFTest,recallCBTest,f1CBTest,accuracyCBTest)
