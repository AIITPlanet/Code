import ReadDataset
import Write_Metric_To_Excel as wte
import numpy as np
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

CV=6
f_NR=1  #Change for respective segmentation
PC_NAME="TIHAS"
#PC_NAME="BTHLAPTOP"
#PC_NAME="HOME"
GivName="5K_nCV_Seg_V2"+str(f_NR)

pathEx="Reports/Results"

#RF Results
precisionRFVal = []
recallRFVal = []
f1RFVal = []
accuracyRFVal= []
precisionRFTest = []
recallRFTest = []
f1RFTest = []
accuracyRFTest= []

#SVM Results 
precisionSVMVal = []
recallSVMVal = []
f1SVMVal = []
accuracySVMVal= []
precisionSVMTest = []
recallSVMTest = []
f1SVMTest = []
accuracySVMTest= []

#CB Results
precisionCBVal = []
recallCBVal = []
f1CBVal = []
accuracyCBVal= []
precisionCBTest = []
recallCBTest = []
f1CBTest = []
accuracyCBTest= []

kFoldCV=[]

path="Excel/combinedDataP_Only_"+str(f_NR)+" - Copy.xlsx"
#path="Excel/combinedDataP5 - Copy.xlsx"

X, Y, Xall, Yall, XBalance,YBalance,XNomatch,YNomatch, XTestBalanced, YTestBalanced =ReadDataset.read(path)

XBalance = np.array(XBalance)  # Convert to NumPy array
YBalance = np.array(YBalance)

XTestBalanced = np.array(XTestBalanced)  # Convert to NumPy array
YTestBalanced = np.array(YTestBalanced)

#-------Unseen data set for test
# XTestBalanced = np.array(XTestBalanced+XNomatch)  # Convert to NumPy array
# YTestBalanced = np.array(YTestBalanced+YNomatch)

for p in range(f_NR):
    #p=4
    XBalancePortion=[]
    YBalancePortion=[]
    XBalancePortionTest=[]
    YBalancePortionTest=[]
    for k in range(int(len(XBalance)/f_NR)):
        XBalancePortion.append(XBalance[k*f_NR+p])
        YBalancePortion.append(YBalance[k*f_NR+p])
        
    XBalancePortion = np.array(XBalancePortion)  # Convert to NumPy array
    YBalancePortion = np.array(YBalancePortion)
    for l in range(int(len(XTestBalanced)/f_NR)):
        XBalancePortionTest.append(XTestBalanced[l*f_NR+p])
        YBalancePortionTest.append(YTestBalanced[l*f_NR+p])
        

    #--------Number of parameters
    print("Reading Training Data set Completed")
    print("Data :",str(len(XBalancePortion)))
    print("Labels :",str(len(YBalancePortion)))
    print("Reading Unseen Data set Completed")
    print("Data :",str(len(XBalancePortionTest)))
    print("Labels :",str(len(YBalancePortionTest)))

    # Split data into training and testing sets
    X_train_Main, X_test_Main, y_train_Main, y_test_Main = train_test_split(XBalancePortion, YBalancePortion, test_size=0.2, random_state=42)
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
                print("Progress_Outer : "+str(i))
                

            # Save the trained model to disk
            joblib.dump(grid_search_RF, './TrainedModels/RF/RF_Neasted_'+GivName+'_'+str(i)+'X'+str(j)+'_'+PC_NAME+'_Portion_'+str(p+1)+'.pkl')
            joblib.dump(grid_search_SVM, './TrainedModels/SVM/SVM_Neasted_'+GivName+'_'+str(i)+'X'+str(j)+'_'+PC_NAME+'_Portion_'+str(p+1)+'.pkl')
            joblib.dump(grid_search_CB, './TrainedModels/CB/CB_Neasted_'+GivName+'_'+str(i)+'X'+str(j)+'_'+PC_NAME+'_Portion_'+str(p+1)+'.pkl')
             # Save the best hyperparameters to a JSON file
            with open('Reports/BestParameters/RF_Neasted_'+str(i)+'X'+str(j)+'_hyperparameters+_Portion_'+str(p+1)+'.json', 'w') as json_file:
                json.dump(best_rf_params, json_file)
            with open('Reports/BestParameters/SVM_Neasted_'+str(i)+'X'+str(j)+'_hyperparameters_Portion_'+str(p+1)+'.json', 'w') as json_file:
                json.dump(best_svm_params, json_file)
            with open('Reports/BestParameters/CD_Neasted_'+str(i)+'X'+str(j)+'_hyperparameters_Portion_'+str(p+1)+'.json', 'w') as json_file:
                json.dump(best_cb_params, json_file)

            kFoldCV.append(str(i)+'X'+str(j))        

            # Calculate and print the classification report for RF
            predicted_classes_RF = grid_search_RF.predict(X_test_Main)
            report_RF = classification_report(y_test_Main, predicted_classes_RF)
            print("Classification Report For RF:")
            print(report_RF)

            # Calculate precision, recall, F1 score, and accuracy
            precisionRFVal.append(precision_score(y_test_Main, predicted_classes_RF, average='macro'))
            recallRFVal.append(recall_score(y_test_Main, predicted_classes_RF, average='macro'))
            f1RFVal.append(f1_score(y_test_Main, predicted_classes_RF, average='macro'))
            accuracyRFVal.append(accuracy_score(y_test_Main, predicted_classes_RF))

            # Calculate and print the classification report for SVM
            predicted_classes_SVM = grid_search_SVM.predict(X_test_Main)
            report_SVM = classification_report(y_test_Main, predicted_classes_SVM)
            print("Classification Report For SVM:")
            print(report_SVM)

            # Calculate precision, recall, F1 score, and accuracy
            precisionSVMVal.append(precision_score(y_test_Main, predicted_classes_SVM, average='macro'))
            recallSVMVal.append(recall_score(y_test_Main, predicted_classes_SVM, average='macro'))
            f1SVMVal.append(f1_score(y_test_Main, predicted_classes_SVM, average='macro'))
            accuracySVMVal.append(accuracy_score(y_test_Main, predicted_classes_SVM))

            # Calculate and print the classification report for SCB
            predicted_classes_CB = grid_search_CB.predict(X_test_Main)
            report_CB = classification_report(y_test_Main, predicted_classes_CB)
            print("Classification Report For CB:")
            print(report_CB)

            # Calculate precision, recall, F1 score, and accuracy
            precisionCBVal.append(precision_score(y_test_Main, predicted_classes_CB, average='macro'))
            recallCBVal.append(recall_score(y_test_Main, predicted_classes_CB, average='macro'))
            f1CBVal.append(f1_score(y_test_Main, predicted_classes_CB, average='macro'))
            accuracyCBVal.append(accuracy_score(y_test_Main, predicted_classes_CB))


            # Calculate and print the classification report for RF on Unseen Data
            predicted_classes_RFU = grid_search_RF.predict(XBalancePortionTest)
            report_RFU = classification_report(YBalancePortionTest, predicted_classes_RFU)
            print("Classification Report For UNSEEN RF:")
            print(report_RFU)

            # Calculate precision, recall, F1 score, and accuracy
            precisionRFTest.append(precision_score(YBalancePortionTest, predicted_classes_RFU, average='macro'))
            recallRFTest.append(recall_score(YBalancePortionTest, predicted_classes_RFU, average='macro'))
            f1RFTest.append(f1_score(YBalancePortionTest, predicted_classes_RFU, average='macro'))
            accuracyRFTest.append(accuracy_score(YBalancePortionTest, predicted_classes_RFU))

            # Calculate and print the classification report for SVM on Unseen Data
            predicted_classes_SVMU = grid_search_SVM.predict(XBalancePortionTest)
            report_SVMU = classification_report(YBalancePortionTest, predicted_classes_SVMU)
            print("Classification Report For UNSEEN SVM:")
            print(report_SVMU)

            # Calculate precision, recall, F1 score, and accuracy
            precisionSVMTest.append(precision_score(YBalancePortionTest, predicted_classes_SVMU, average='macro'))
            recallSVMTest.append(recall_score(YBalancePortionTest, predicted_classes_SVMU, average='macro'))
            f1SVMTest.append(f1_score(YBalancePortionTest, predicted_classes_SVMU, average='macro'))
            accuracySVMTest.append(accuracy_score(YBalancePortionTest, predicted_classes_SVMU))

            # Calculate and print the classification report for CB on Unseen Data
            predicted_classes_CBU = grid_search_CB.predict(XBalancePortionTest)
            report_CBU = classification_report(YBalancePortionTest, predicted_classes_CBU)
            print("Classification Report For UNSEEN CB:")
            print(report_CBU)

            # Calculate precision, recall, F1 score, and accuracy
            precisionCBTest.append(precision_score(YBalancePortionTest, predicted_classes_CBU, average='macro'))
            recallCBTest.append(recall_score(YBalancePortionTest, predicted_classes_CBU, average='macro'))
            f1CBTest.append(f1_score(YBalancePortionTest, predicted_classes_CBU, average='macro'))
            accuracyCBTest.append(accuracy_score(YBalancePortionTest, predicted_classes_CBU))

          
            #print(str(x.strftime("%Y")+x.strftime("%m")+x.strftime("%d")+x.strftime("%H")+"_data.txt"))
            outfile=open("Reports/Nested_Reports/Validation_"+GivName+"_"+str(f_NR)+"_"+PC_NAME+"_Portion_"+str(p+1)+"_data.txt", "a")
            outfile.write("Nested : "+str(i)+"X"+str(j)+" : \nRF :\n"+report_RF+"\nSVM :\n"+report_SVM+"\nCB :\n"+report_CB)
            outfile.write('\n')
            outfile.close()
            #print(str(x.strftime("%Y")+x.strftime("%m")+x.strftime("%d")+x.strftime("%H")+"_data.txt"))
            outfile=open("Reports/Nested_Reports/Test_"+GivName+"_"+str(f_NR)+"_"+PC_NAME+"_Portion_"+str(p+1)+"_data.txt", "a")
            outfile.write("Nested :"+str(i)+"X"+str(j)+" : \nRF :\n"+report_RFU+"\nSVM :\n"+report_SVMU+"\nCB :\n"+report_CBU)
            outfile.write('\n')
            outfile.close()
    
    wte.WriteToExcel(pathEx,GivName+"_P_"+str(p+1)+"_RFVal"+PC_NAME,kFoldCV,precisionRFVal,recallRFVal,f1RFVal,accuracyRFVal)
    wte.WriteToExcel(pathEx,GivName+"_P_"+str(p+1)+"_RFTest"+PC_NAME,kFoldCV,precisionRFTest,recallRFTest,f1RFTest,accuracyRFTest)

    wte.WriteToExcel(pathEx,GivName+"_P_"+str(p+1)+"_SVMVal"+PC_NAME,kFoldCV,precisionSVMVal,recallSVMVal,f1SVMVal,accuracySVMVal)
    wte.WriteToExcel(pathEx,GivName+"_P_"+str(p+1)+"_SVMTest"+PC_NAME,kFoldCV,precisionSVMTest,recallSVMTest,f1SVMTest,accuracySVMTest)

    wte.WriteToExcel(pathEx,GivName+"_P_"+str(p+1)+"_CBVal"+PC_NAME,kFoldCV,precisionCBVal,recallCBVal,f1CBVal,accuracyCBVal)
    wte.WriteToExcel(pathEx,GivName+"_P_"+str(p+1)+"_CBTest"+PC_NAME,kFoldCV,precisionRFTest,recallCBTest,f1CBTest,accuracyCBTest)
