import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler, StandardScaler

def are_all_integers(arr):
    for num in arr:
        if not isinstance(num, int): #check if the number is integer
            if not (isinstance(num, float) and num.is_integer()): #number is a float with no decimal part
                return False
    return True

def scale(data,method,scale_only_continuous=True,scale_everything=False,drop_correlated_cols=True):
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    
    scaled_data = data.copy()
    if drop_correlated_cols == True:
        scaled_data.drop(['totaldaycharge', 'totalevecharge', 'totalnightcharge', 'totalintlcharge'], axis=1, inplace=True)

    for col in scaled_data.columns:
        if col not in ["churn", "internationalplan", "voicemailplan"] and method != "none":
            if scale_everything == True:  #scales eveything except the binary categoric variables
                scaled_data.loc[:, col] = scaler.fit_transform(scaled_data[[col]])
            elif scale_only_continuous == True: #scales only the continuous variables
                continuous_col = are_all_integers(scaled_data[[col]])
                if continuous_col == True:
                    scaled_data.loc[:, col] = scaler.fit_transform(scaled_data[[col]])
        elif col not in ["internationalplan","voicemailplan"]:
            scaled_data.replace(['No', 'Yes'],[0, 1], inplace=True)
        else:
            scaled_data.replace(['no', 'yes'],[0, 1], inplace=True)

    return scaled_data
    

def PCA_explained_variance_plots(data_without_nan):
    numeric_features = ["accountlength", "numbervmailmessages", "totaldayminutes", "totaldaycalls", "totaldaycharge",
                        "totaleveminutes", "totalevecalls", "totalevecharge", "totalnightminutes", "totalnightcalls",
                        "totalnightcharge", "totalintlminutes", "totalintlcalls", "totalintlcharge","numbercustomerservicecalls"]
    combinations = ["Standard scaling with all variables", "Minmax scaling with all variables","Unscaled with all variables",
           "Standard scaling without correlated variables", "Minmax scaling without correlated variables","Unscaled without correlated variables"]
    methods = ["standard","minmax","none","standard","minmax","none"]
    drop_correlated_cols_list = [False, False, False, True, True, True]
    scale_everything_list = [True, True, False, True, True, False]
    n_components_list=[15,15,15,11,11,11]
    fig1, ax1 = plt.subplots(2,3,figsize=(14,7))
    for i,comb in enumerate(combinations):
        #print(comb)
        if i <= 2: n = 0
        else: n = 1
        if i<= 2: m = i
        else: m=i-3
        data=data_without_nan.copy()
        scaled_data = scale(data,method=methods[i],scale_only_continuous=False,
                            scale_everything=scale_everything_list[i],drop_correlated_cols=drop_correlated_cols_list[i])
        pca = PCA(n_components=n_components_list[i])
        if drop_correlated_cols_list[i] == True:
            cols_to_drop = ['totaldaycharge', 'totalevecharge', 'totalnightcharge', 'totalintlcharge']
            numeric_features = [x for x in numeric_features if x not in cols_to_drop]
        x = scaled_data.loc[:, numeric_features].values
        principalComponents = pca.fit_transform(x)
        principal_Df = pd.DataFrame(data=principalComponents, columns=[f'principal component {i}' for i in range(1, n_components_list[i] + 1)])
        
        churn = scaled_data["churn"].reset_index() #add 'churn' column to principal_Df
        del churn['index']
        finalDf = pd.concat([principal_Df, churn], axis=1, ignore_index=False)

        #explained variance ratio
        explained_variance_ratio = pca.explained_variance_ratio_
        ax1[n, m].bar(range(1, n_components_list[i] + 1), explained_variance_ratio, alpha=0.7, align='center', label='Explained Variance Ratio')
        ax1[n, m].step(range(1, n_components_list[i] + 1), np.cumsum(explained_variance_ratio), where='mid', label='Cumulative Explained Variance')
        ax1[n, m].set_xlabel('Principal Component Index', fontsize=7)
        ax1[n, m].set_ylabel('Explained Variance Ratio', fontsize=7)
        ax1[n, m].set_title(combinations[i], fontsize=10)
    fig1.suptitle("Explained variance of Principal Components")
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig("PCA_explained_variance.png", bbox_inches='tight')


def PCA_fit(scaled_data, drop_correlated_cols, n_components,print_explained_variance):
    numeric_features = ["accountlength", "numbervmailmessages", "totaldayminutes", "totaldaycalls", "totaldaycharge",
                        "totaleveminutes", "totalevecalls", "totalevecharge", "totalnightminutes", "totalnightcalls",
                        "totalnightcharge", "totalintlminutes", "totalintlcalls", "totalintlcharge","numbercustomerservicecalls"]
    
    if drop_correlated_cols == True:
        cols_to_drop = ['totaldaycharge', 'totalevecharge', 'totalnightcharge', 'totalintlcharge']
        numeric_features = [x for x in numeric_features if x not in cols_to_drop]
        
    pca = PCA(n_components=n_components)
    x = scaled_data.loc[:, numeric_features].values
    principalComponents = pca.fit_transform(x)
    principal_Df = pd.DataFrame(data=principalComponents, columns=[f'principal component {i}' for i in range(1, n_components + 1)])
    
    churn = scaled_data["churn"].reset_index() #add 'churn' column to principal_Df
    del churn['index']
    finalDf = pd.concat([principal_Df, churn], axis=1, ignore_index=False)

    if n_components == 15:
        plt.figure(figsize=(9, 9))
    elif n_components == 11:
        plt.figure(figsize=(8, 8))
    elif n_components == 2:
        plt.figure(figsize=(3,3))
    
    for i in range(1, n_components):
        if n_components == 15:
            plt.subplot(5, 3, i)
        elif n_components == 11:
            plt.subplot(4, 3, i)
        plt.xticks(fontsize=6); plt.yticks(fontsize=5)
        plt.xlabel(f'Principal Component {i}', fontsize=8)
        plt.ylabel(f'Principal Component {i + 1}', fontsize=8)
    
        for target, color in zip([0, 1], ['r', 'g']):
            indicesToKeep = finalDf['churn'] == target
            plt.scatter(finalDf.loc[indicesToKeep, f'principal component {i}'],
                        finalDf.loc[indicesToKeep, f'principal component {i + 1}'],
                        c=color, s=6, label=target)
    
        plt.legend(['No', 'Yes'], prop={'size': 6})

    plt.suptitle("Principal Component Analysis of Dataset", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("PCA_standard_corr_drop.png", bbox_inches='tight')
    plt.show()

    #explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    if print_explained_variance == True:
        for i in range(1, n_components + 1):
            print(f"Principal Component {i}: {explained_variance_ratio[i - 1]:.4f}")

    plt.figure(figsize=(5, 3))
    plt.bar(range(1, n_components + 1), explained_variance_ratio, alpha=0.7, align='center', label='Explained Variance Ratio')
    plt.step(range(1, n_components + 1), np.cumsum(explained_variance_ratio), where='mid', label='Cumulative Explained Variance')
    plt.xlabel('Principal Component Index', fontsize=7); plt.ylabel('Explained Variance Ratio', fontsize=7)
    plt.title('Explained Variance of Principal Components', fontsize=10); plt.legend(loc='best', fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("PCA_Variance_standard_corr_drop.png", bbox_inches='tight')
    plt.show()