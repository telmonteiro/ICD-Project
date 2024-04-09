from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

def model_fit_pred(model, scaled_data,X_train,Y_train,X_test,Y_test):
    if isinstance(scaled_data, str) == False:
        X_train, X_test, Y_train, Y_test = train_test_split(scaled_data.values[:,1:], scaled_data['churn'].values.reshape(len(scaled_data["churn"]),), 
                                                        train_size=0.8, test_size=0.2,shuffle=False)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    classification_rep = classification_report(Y_test, y_pred,digits=5,output_dict=True,zero_division=1)
    model_roc_auc=roc_auc_score(Y_test,y_pred)
    return accuracy, classification_rep, model_roc_auc

def lda_model(scaled_data,cross_validation,num_folds,solv,shrinkage_type):
    if cross_validation == True:
        X_data = scaled_data.values[:,1:]
        Y_data = scaled_data['churn'].values.reshape(len(scaled_data["churn"]),)
        lda = LinearDiscriminantAnalysis()
        if solv=="svd":
            param_grid = dict(solver=["svd"])
            grid = GridSearchCV(lda, param_grid, scoring=["accuracy","f1","f1_weighted"], n_jobs=None, refit="f1",
                                cv=num_folds, verbose=0, return_train_score=True)
            grid.fit(X_data,Y_data)
            results = grid.cv_results_
            model_best = grid.best_estimator_
            accuracy, classification_rep, model_roc_auc = model_fit_pred(model_best,scaled_data=scaled_data,X_train=None,Y_train=None,X_test=None,Y_test=None)
            print("Best Model: ", grid.best_estimator_)
            accuracy_list_reg = results["mean_test_accuracy"]
            f1score_1_list_reg = results["mean_test_f1"]
            f1score_list_reg =  results["mean_test_f1_weighted"]
            
            precision = classification_rep['weighted avg']["precision"]
            f1score_1 = classification_rep['1']['f1-score']
            f1score = classification_rep['weighted avg']['f1-score']            
            print("SVD Model")
            print("F1-Score for 1: ",round(f1score_1,3))
            print("Accuracy: ", round(accuracy,3))
            print("Weighted F1-Score: ",round(f1score,3))
            print("Precision: ", round(precision,3))
            print("ROC AUC: ", round(model_roc_auc,3))
            print("------------------------------------------------------------")
            return accuracy_list_reg, classification_rep, f1score_list_reg, f1score_1_list_reg, model_roc_auc, precision
        else:
            if shrinkage_type == "normal":
                shrink_list= np.arange(0,1,0.001)
                if solv == "lsqr":    
                    param_grid = dict(solver=[solv],shrinkage=shrink_list)
                    grid = GridSearchCV(lda, param_grid, scoring=["accuracy","f1","f1_weighted"], n_jobs=None, refit="f1",
                                        cv=num_folds, verbose=0, return_train_score=True)
                    grid.fit(X_data,Y_data)
                    results = grid.cv_results_
                    model_best = grid.best_estimator_
                    print("Best Model: ", grid.best_estimator_)
                    accuracy, classification_rep, model_roc_auc = model_fit_pred(model_best,scaled_data=scaled_data,
                                                                                 X_train=None,Y_train=None,X_test=None,Y_test=None)
                    accuracy_list_reg = results["mean_test_accuracy"]
                    f1score_1_list_reg = results["mean_test_f1"]
                    f1score_list_reg =  results["mean_test_f1_weighted"]
                    
                    precision = classification_rep['weighted avg']["precision"]
                    f1score_1 = classification_rep['1']['f1-score']
                    f1score = classification_rep['weighted avg']['f1-score']
                
                    print("F1-Score for 1: ",round(f1score_1,3))
                    print("Accuracy: ", round(accuracy,3))
                    print("Weighted F1-Score: ",round(f1score,3))
                    print("Precision: ", round(precision,3))
                    print("ROC AUC: ", round(model_roc_auc,3))
                    print("------------------------------------------------------------")
                    return accuracy_list_reg, classification_rep, f1score_list_reg, f1score_1_list_reg, model_roc_auc, precision
                else:
                    param_grid = dict(solver=[solv],shrinkage=shrink_list)
                    grid = GridSearchCV(lda, param_grid, scoring=["accuracy","f1","f1_weighted"], n_jobs=None, refit="f1",
                                        cv=num_folds, verbose=0, return_train_score=True)
                    grid.fit(X_data,Y_data)
                    results = grid.cv_results_
                    model_best = grid.best_estimator_
                    print("Best Model: ", grid.best_estimator_)
                    accuracy, classification_rep, model_roc_auc = model_fit_pred(model_best,scaled_data=scaled_data,
                                                                                 X_train=None,Y_train=None,X_test=None,Y_test=None)
                    accuracy_list_reg = results["mean_test_accuracy"]
                    f1score_1_list_reg = results["mean_test_f1"]
                    f1score_list_reg =  results["mean_test_f1_weighted"]
                    
                    precision = classification_rep['weighted avg']["precision"]
                    f1score_1 = classification_rep['1']['f1-score']
                    f1score = classification_rep['weighted avg']['f1-score']
                    print("F1-Score for 1: ",round(f1score_1,3))
                    print("Accuracy: ", round(accuracy,3))
                    print("Weighted F1-Score: ",round(f1score,3))
                    print("Precision: ", round(precision,3))
                    print("ROC AUC: ", round(model_roc_auc,3))
                    print("------------------------------------------------------------")
                    return accuracy_list_reg, classification_rep, f1score_list_reg, f1score_1_list_reg, model_roc_auc, precision
            else:
                if solv == "lsqr":
                    param_grid = dict(solver=[solv],shrinkage=["auto"])
                    grid = GridSearchCV(lda, param_grid, scoring=["accuracy","f1","f1_weighted"], n_jobs=None, refit="f1",
                                        cv=num_folds, verbose=0, return_train_score=True)
                    grid.fit(X_data,Y_data)
                    results = grid.cv_results_
                    model_best = grid.best_estimator_
                    print("Best Model: ", grid.best_estimator_)
                    accuracy, classification_rep, model_roc_auc = model_fit_pred(model_best,scaled_data=scaled_data,
                                                                                 X_train=None,Y_train=None,X_test=None,Y_test=None)
                    ### the best model is chosen based on the F1-score for 1
                    f1score_1 = classification_rep['1']['f1-score']
                    f1score = classification_rep['weighted avg']['f1-score']
                    precision = classification_rep["weighted avg"]["precision"]
                    print("F1-Score for 1: ",round(f1score_1,3))
                    print("Accuracy: ", round(accuracy,3))
                    print("Weighted F1-Score: ",round(f1score,3))
                    print("Precision: ", round(precision,3))
                    print("ROC AUC: ", round(model_roc_auc,3))
                    print("------------------------------------------------------------")
                    accuracy_list_reg = results["mean_test_accuracy"]
                    f1score_1_list_reg = results["mean_test_f1"]
                    f1score_list_reg =  results["mean_test_f1_weighted"]
                    return accuracy_list_reg, classification_rep, f1score_list_reg, f1score_1_list_reg, model_roc_auc, precision
                else:
                    param_grid = dict(solver=[solv],shrinkage=["auto"])
                    grid = GridSearchCV(lda, param_grid, scoring=["accuracy","f1","f1_weighted"], n_jobs=None, refit="f1",
                                        cv=num_folds, verbose=0, return_train_score=True)
                    grid.fit(X_data,Y_data)
                    results = grid.cv_results_
                    model_best = grid.best_estimator_
                    print("Best Model: ", grid.best_estimator_)
                    accuracy, classification_rep, model_roc_auc = model_fit_pred(model_best,scaled_data=scaled_data,
                                                                                 X_train=None,Y_train=None,X_test=None,Y_test=None)
                    ### the best model is chosen based on the F1-score for 1
                    f1score_1 = classification_rep['1']['f1-score']
                    f1score = classification_rep['weighted avg']['f1-score']
                    precision = classification_rep["weighted avg"]["precision"]
                    print("F1-Score for 1: ",round(f1score_1,3))
                    print("Accuracy: ", round(accuracy,3))
                    print("Weighted F1-Score: ",round(f1score,3))
                    print("Precision: ", round(precision,3))
                    print("ROC AUC: ", round(model_roc_auc,3))
                    print("------------------------------------------------------------")
                    accuracy_list_reg = results["mean_test_accuracy"]
                    f1score_1_list_reg = results["mean_test_f1"]
                    f1score_list_reg =  results["mean_test_f1_weighted"]
                    return accuracy_list_reg, classification_rep, f1score_list_reg, f1score_1_list_reg, model_roc_auc, precision

    elif cross_validation == False:
        shrink_list= np.arange(0,1,0.001)
        X_train, X_test, Y_train, Y_test = train_test_split(scaled_data.values[:,1:], scaled_data['churn'].values.reshape(len(scaled_data["churn"]),), 
                                                        train_size=0.8, test_size=0.2,shuffle=False)
        if solv == "svd":
            lda = LinearDiscriminantAnalysis(solver=solv)
            accuracy,classification_rep, model_roc_auc = model_fit_pred(lda,scaled_data="null",X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test)
            f1score = classification_rep['weighted avg']['f1-score']
            f1score_1 = classification_rep["1"]["f1-score"]
            precision = classification_rep['weighted avg']["precision"]
            print("SVD Model")
            print("F1-Score for 1: ",round(f1score_1,3))
            print("Accuracy: ", round(accuracy,3))
            print("Weighted F1-Score: ",round(f1score,3))
            print("Precision: ", round(precision,3))
            print("ROC AUC: ", round(model_roc_auc,3))
            print("------------------------------------------------------------")
            return accuracy, classification_rep, f1score, f1score_1, model_roc_auc, precision
        else:
            if shrinkage_type == "normal":
                accuracy_list_shrink = []; f1score_list_shrink=[]; f1score_1_list_shrink = []; precision_list = []; roc_list = []
                for shrink in shrink_list:
                    lda = LinearDiscriminantAnalysis(solver=solv,shrinkage=shrink)
                    accuracy,classification_rep, model_roc_auc = model_fit_pred(lda,scaled_data="null",
                                                                                X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test)
                    f1score = classification_rep['weighted avg']['f1-score']
                    f1score_1 = classification_rep["1"]["f1-score"]
                    precision = classification_rep["weighted avg"]["precision"]
                    accuracy_list_shrink.append(accuracy)
                    f1score_list_shrink.append(f1score)
                    f1score_1_list_shrink.append(f1score_1)
                    precision_list.append(precision)
                    roc_list.append(model_roc_auc)

                ### the best model is chosen based on the F1-score for 1
                max_f1_1_index = f1score_1_list_shrink.index(max(f1score_1_list_shrink))
                max_f1_1_shrink = shrink_list[max_f1_1_index]
                
                lda = LinearDiscriminantAnalysis(solver=solv,shrinkage=max_f1_1_shrink)
                accuracy, classification_rep, model_roc_auc = model_fit_pred(lda,scaled_data="null",
                                                                             X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test)
                f1score_1 = classification_rep['1']['f1-score']
                f1score = classification_rep['weighted avg']['f1-score']
                precision = classification_rep["weighted avg"]["precision"]
                print("Best {} model shrinkage: ".format(solv),round(max_f1_1_shrink,3))
                print("F1-Score for 1: ",round(f1score_1,3))
                print("Accuracy: ", round(accuracy,3))
                print("Weighted F1-Score: ",round(f1score,3))
                print("Precision: ", round(precision,3))
                print("ROC AUC: ", round(model_roc_auc,3))
                print("------------------------------------------------------------")
                return accuracy_list_shrink, classification_rep, f1score_list_shrink,f1score_1_list_shrink, model_roc_auc, precision
                
            elif shrinkage_type == "auto":
                lda = LinearDiscriminantAnalysis(solver=solv,shrinkage="auto")
                accuracy,classification_rep, model_roc_auc = model_fit_pred(lda,scaled_data="null",X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test)
                f1score = classification_rep['weighted avg']['f1-score']
                f1score_1 = classification_rep['1']['f1-score']
                precision = classification_rep["weighted avg"]["precision"]
                
                print("Best {} auto model".format(solv))
                print("F1-Score for 1: ",round(f1score_1,3))
                print("Accuracy: ", round(accuracy,3))
                print("Weighted F1-Score: ",round(f1score,3))
                print("Precision: ", round(precision,3))
                print("ROC AUC: ", round(model_roc_auc,3))
                print("------------------------------------------------------------")
                
                return accuracy, classification_rep, f1score,f1score_1, model_roc_auc, precision

def qda_model(scaled_data,cross_validation,num_folds):
    reg_param_list= np.arange(0,1,0.001)
    
    if cross_validation == True:
        X_data = scaled_data.values[:,1:]
        Y_data = scaled_data['churn'].values.reshape(len(scaled_data["churn"]),)
        qda = QuadraticDiscriminantAnalysis()
        param_grid = dict(reg_param=reg_param_list)
        grid = GridSearchCV(qda, param_grid, scoring=["accuracy","f1","f1_weighted"], n_jobs=None, refit="f1",
                            cv=num_folds, verbose=0, return_train_score=True)
        grid.fit(X_data,Y_data)
        results = grid.cv_results_
        model_best = grid.best_estimator_
        print("Best QDA Model: ", grid.best_estimator_)
        accuracy, classification_rep, model_roc_auc = model_fit_pred(model_best,scaled_data=scaled_data,X_train=None,Y_train=None,X_test=None,Y_test=None)
        accuracy_list_reg = results["mean_test_accuracy"]
        f1score_1_list_reg = results["mean_test_f1"]
        f1score_list_reg = results["mean_test_f1_weighted"]
        
        f1score = classification_rep['weighted avg']['f1-score']
        f1score_1 = classification_rep['1']['f1-score']
        precision = classification_rep["weighted avg"]["precision"]
        print("F1-Score for 1: ",round(f1score_1,3))
        print("Accuracy: ", round(accuracy,3))
        print("Weighted F1-Score: ",round(f1score,3))
        print("Precision: ", round(precision,3))
        print("ROC AUC: ", round(model_roc_auc,3))
        print("------------------------------------------------------------")
        
        return accuracy_list_reg, classification_rep, f1score_list_reg, f1score_1_list_reg, model_roc_auc, precision

    elif cross_validation == False:
        X_train, X_test, Y_train, Y_test = train_test_split(scaled_data.values[:,1:], scaled_data['churn'].values.reshape(len(scaled_data["churn"]),), 
                                                        train_size=0.8, test_size=0.2,shuffle=False)
        
        accuracy_list_reg = []; f1score_list_reg = []; f1score_1_list_reg = []; roc_list = []; precision_list=[]
        for reg in reg_param_list:
            qda = QuadraticDiscriminantAnalysis(reg_param=reg)
            accuracy,classification_rep, model_roc_auc = model_fit_pred(qda,scaled_data="null",X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test)
            
            if '0.0' in classification_rep and classification_rep['0.0']['precision'] == 0.0:
                # Handle the case where precision is ill-defined (e.g., set it to a specific value)
                classification_rep['0.0']['precision'] = 0.0
                
            accuracy_list_reg.append(accuracy)
            f1score_list_reg.append(classification_rep['weighted avg']['f1-score'])
            f1score_1_list_reg.append(classification_rep['1']['f1-score'])
            roc_list.append(model_roc_auc)
            precision_list.append(classification_rep["weighted avg"]["precision"])

        #choosing best model based on F1-score for 1
        max_f1_1_index = f1score_1_list_reg.index(max(f1score_1_list_reg))
        max_f1_1_reg = reg_param_list[max_f1_1_index]
        qda = QuadraticDiscriminantAnalysis(reg_param=max_f1_1_reg)
        accuracy,classification_rep,model_roc_auc = model_fit_pred(qda,scaled_data="null",X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test)
        if '0.0' in classification_rep and classification_rep['0.0']['precision'] == 0.0:
            # Handle the case where precision is ill-defined (e.g., set it to a specific value)
            classification_rep['0.0']['precision'] = 0.0
        
        f1score_1 = classification_rep['1']['f1-score']
        f1score = classification_rep['weighted avg']['f1-score']
        precision = classification_rep["weighted avg"]["precision"]
        print("Best QDA model reg: ",round(max_f1_1_reg,3))
        print("F1-Score for 1: ",round(f1score_1,3))
        print("Accuracy: ", round(accuracy,3))
        print("Weighted F1-Score: ",round(f1score,3))
        print("Precision: ", round(precision,3))
        print("ROC AUC: ", round(model_roc_auc,3))
        print("------------------------------------------------------------")

        return accuracy_list_reg, classification_rep, f1score_list_reg, f1score_1_list_reg, model_roc_auc, precision