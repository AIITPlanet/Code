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

#nCV combinations
CV=11 #The combinatins CV=11 will run up to CV -1 for i,j=10

#full Sequnce or segment selection
f_NR=1

#Machine identifier
PC_NAME="TIHASERVER"
#PC_NAME="BTHLAPTOP"
#PC_NAME="HOME"

#Output location
pathEx="Reports"

#Metrics to store
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


#Dataset Location and Name
path="Excel/combinedDataP_Only_"+str(f_NR)+" - Copy.xlsx"
#path="Excel/combinedDataP5 - Copy.xlsx"

#Participant level data distribution isolates XTestBalanced, YTestBalanced for filnal test and XBalance, YBalance for training and validation set.
#X, Y, Xall, Yall, XBalance,YBalance,XNomatch,YNomatch, XTestBalanced, YTestBalanced =ReadDataset.read(path)
X, Y, _, _, XBalance, YBalance, _, _, XTestBalanced, YTestBalanced = ReadDataset.read(path)

#remaining data
XBalance = np.array(XBalance)  # Convert to NumPy array
YBalance = np.array(YBalance)

#Isolated data test set
XTestBalanced = np.array(XTestBalanced)  # Convert to NumPy array
YTestBalanced = np.array(YTestBalanced)


#--------Number of parameters
print("Reading Training Data set Completed")
print("Data :",str(len(XBalance)))
print("Labels :",str(len(YBalance)))
print("Reading Unseen Data set Completed")
print("Data :",str(len(XTestBalanced)))
print("Labels :",str(len(YTestBalanced)))

# Split data into training and validation set sets
X_train_Main, X_test_Main, y_train_Main, y_test_Main = train_test_split(XBalance, YBalance, test_size=0.2, random_state=42)
print("Training dataset splited to two sections")
#--------Number of parameters
print("Spliting Training Data set Completed")
print("Data :",str(len(X_train_Main)))
print("Labels :",str(len(y_train_Main)))
print("Spliting Validation Data set Completed")
print("Data :",str(len(X_test_Main)))
print("Labels :",str(len(y_test_Main)))

#nCV combinations

for i in range(2,CV):
    for j in range(2,CV):
        print("Nested_outer_inner_"+str(i)+" X "+str(j))

        # Create all classifiers for each nCV combination
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
        #outer cross-validation loop for hyperparameter tuning
        outer_cv = KFold(n_splits=i, shuffle=True, random_state=42)
        # Lists to store RF the results
        outer_scores_RF= []
        # Lists to store SVM the results
        outer_scores_SVM = []
        # Lists to store CB the results
        outer_scores_CB = []
        #spliting outer loop on trining data
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
            print("Progress_RF : "+str(outer_score_RF))
            print("Progress_SVM : "+str(outer_score_SVM))
            print("Progress_CB : "+str(outer_score_CB))

        # Save the trained model to disk for later analysis
        joblib.dump(grid_search_RF, './TrainedModels/Analyse_RF_Neasted_'+str(i)+'X'+str(j)+'_'+PC_NAME+'.pkl')
        joblib.dump(grid_search_SVM, './TrainedModels/Analyse_SVM_Neasted_'+str(i)+'X'+str(j)+'_'+PC_NAME+'.pkl')
        joblib.dump(grid_search_CB, './TrainedModels/Analyse_CB_Neasted_'+str(i)+'X'+str(j)+'_'+PC_NAME+'.pkl')

        #Save nCV combination
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
        recallCBVal.append(recall_score(y_test_Main, predicted_classes_CB, average='wmacro'))
        f1CBVal.append(f1_score(y_test_Main, predicted_classes_CB, average='macro'))
        accuracyCBVal.append(accuracy_score(y_test_Main, predicted_classes_CB))


        # Calculate and print the classification report for RF on Unseen Data
        predicted_classes_RFU = grid_search_RF.predict(XTestBalanced)
        report_RFU = classification_report(YTestBalanced, predicted_classes_RFU)
        print("Classification Report For UNSEEN RF:")
        print(report_RFU)

         # Calculate precision, recall, F1 score, and accuracy
        precisionRFTest.append(precision_score(YTestBalanced, predicted_classes_RFU, average='macro'))
        recallRFTest.append(recall_score(YTestBalanced, predicted_classes_RFU, average='macro'))
        f1RFTest.append(f1_score(YTestBalanced, predicted_classes_RFU, average='macro'))
        accuracyRFTest.append(accuracy_score(YTestBalanced, predicted_classes_RFU))

        # Calculate and print the classification report for SVM on Unseen Data
        predicted_classes_SVMU = grid_search_SVM.predict(XTestBalanced)
        report_SVMU = classification_report(YTestBalanced, predicted_classes_SVMU)
        print("Classification Report For UNSEEN SVM:")
        print(report_SVMU)

         # Calculate precision, recall, F1 score, and accuracy
        precisionSVMTest.append(precision_score(YTestBalanced, predicted_classes_SVMU, average='macro'))
        recallSVMTest.append(recall_score(YTestBalanced, predicted_classes_SVMU, average='macro'))
        f1SVMTest.append(f1_score(YTestBalanced, predicted_classes_SVMU, average='macro'))
        accuracySVMTest.append(accuracy_score(YTestBalanced, predicted_classes_SVMU))

        # Calculate and print the classification report for CB on Unseen Data
        predicted_classes_CBU = grid_search_CB.predict(XTestBalanced)
        report_CBU = classification_report(YTestBalanced, predicted_classes_CBU)
        print("Classification Report For UNSEEN CB:")
        print(report_CBU)

         # Calculate precision, recall, F1 score, and accuracy
        precisionCBTest.append(precision_score(YTestBalanced, predicted_classes_CBU, average='macro'))
        recallCBTest.append(recall_score(YTestBalanced, predicted_classes_CBU, average='wmacro'))
        f1CBTest.append(f1_score(YTestBalanced, predicted_classes_CBU, average='macro'))
        accuracyCBTest.append(accuracy_score(YTestBalanced, predicted_classes_CBU))


        #Write all Validation set reports into a textfile
        #print(str(x.strftime("%Y")+x.strftime("%m")+x.strftime("%d")+x.strftime("%H")+"_data.txt"))
        outfile=open("Reports/Nested_Reports/Validation_"+str(f_NR)+"_"+PC_NAME+"_data.txt", "a")
        outfile.write("Nested : "+str(i)+"X"+str(j)+" : \nRF :\n"+report_RF+"\nSVM :\n"+report_SVM+"\nCB :\n"+report_CB)
        outfile.write('\n')
        outfile.close()
        #Write all Test set reports into a textfile
        #print(str(x.strftime("%Y")+x.strftime("%m")+x.strftime("%d")+x.strftime("%H")+"_data.txt"))
        outfile=open("Reports/Nested_Reports/Test_"+str(f_NR)+"_"+PC_NAME+"_data.txt", "a")
        outfile.write("Nested :"+str(i)+"X"+str(j)+" : \nRF :\n"+report_RFU+"\nSVM :\n"+report_SVMU+"\nCB :\n"+report_CBU)
        outfile.write('\n')
        outfile.close()
        kFoldCV.append(str(i)+"X"+str(j))
#Write all Validation and Test metrics into Excel file
wte.WriteToExcel(pathEx,"RFVal"+PC_NAME,kFoldCV,precisionRFVal,recallRFVal,f1RFVal,accuracyRFVal)
wte.WriteToExcel(pathEx,"RFTest"+PC_NAME,kFoldCV,precisionRFTest,recallRFTest,f1RFTest,accuracyRFTest)

wte.WriteToExcel(pathEx,"SVMVal"+PC_NAME,kFoldCV,precisionSVMVal,recallSVMVal,f1SVMVal,accuracySVMVal)
wte.WriteToExcel(pathEx,"SVMTest"+PC_NAME,kFoldCV,precisionSVMTest,recallSVMTest,f1SVMTest,accuracySVMTest)

wte.WriteToExcel(pathEx,"CBVal"+PC_NAME,kFoldCV,precisionCBVal,recallCBVal,f1CBVal,accuracyCBVal)
wte.WriteToExcel(pathEx,"CBTest"+PC_NAME,kFoldCV,precisionRFTest,recallCBTest,f1CBTest,accuracyCBTest)


