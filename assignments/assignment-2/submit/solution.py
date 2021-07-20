#!/usr/bin/env python
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Fill in your names using the format below
your_name = "Beeren, Nimo"


# # Machine Learning Engineering - Assignment 2
# The [Speed Dating dataset](https://www.openml.org/d/40536) collects feedback gathered from participants in experimental speed dating events. Every participant rated themselves and their dates according to different attributes (e.g. attractiveness, sincerity, intelligence, fun, ambition, and shared interests,...), and whether or not they were interested in a second date. Our goal is to build a machine learning model able to predict whether there will be a match (or not) between two different people. Will you be able to trust your final model?

# In[2]:


# imports
# get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openml
import time
start_time = time.time()


# ### Additional packages:
# * Seaborn (plotting)
#     - Run `pip install seaborn` or `conda install seaborn`
# * tqdm (progress bar for longer running jobs)
#     - Run `pip install tqdm`

# In[3]:


# Pre-flight checklist. Do not change this code.
# Make sure that you have installed recent versions of key packages.
# You could lose points if these checks do not pass.
from packaging import version
import sklearn
import seaborn as sns
from tqdm.notebook import tqdm
sklearn_version = sklearn.__version__
if version.parse(sklearn_version) < version.parse("0.24.0"):
    print("scikit-learn is outdated. Please update now! pip install -U scikit-learn")
else:
    print("OK. You may continue :)")


# In[4]:


# Download Speed Dating data. Takes a while the first time. Do not change this code!
# Note that X is a pandas dataframe 
dates = openml.datasets.get_dataset(40536)
X, y, _, feat_names = dates.get_data(target=dates.default_target_attribute)


# In[5]:


# Cleanup. Do not change this code!
# Remove irrelevant or preprocessed columns 
cols = [c for c in X.columns if ((c.lower()[:2] != 'd_' or c.lower() == 'd_age') and c.lower() not in ['wave','has_null'])]
X = X[cols]
classes = ['No match','Match']

# Fix feature name typos
X = X.rename(columns={'ambtition_important': 'ambition_important',
                     'sinsere_o': 'sincere_o'})

# Harmonize the field names somewhat
X['field'] = X['field'].str.lower()
X = X.astype({'field': 'category'})

# Drop columns with more than 10% missing values
missing_counts = X.isnull().sum() * 100 / len(X)
d = {k:v for (k,v) in missing_counts.items() if v>10}
X.drop(d.keys(), axis=1, inplace=True)

# Solves an implementation issue with TargetEncoder
y=y.astype(int)


# In[6]:


# THIS WILL BE HELPFUL, but do not use these as global variables in your functions!
# The list of the names of all categorical features
categorical = X.select_dtypes(include=["category"]).columns.tolist()
# The list of the names of all numerical features
numerical = X.select_dtypes(exclude=["category"]).columns.tolist()


# In[7]:


# Peek at the remaining data
X


# In[8]:


# Check the column data types and missing data
X.info()


# In[9]:


# Is the data balanced?
y.value_counts().plot(kind='barh', figsize=(5,2));


# In[10]:


# See if some groups of people are more represented than others
X['gender'].value_counts().plot(kind='barh', figsize=(5,2));


# In[11]:


X['race'].value_counts().plot(kind='barh', figsize=(5,2));


# In[12]:


# Some categorical columns have a large number of possible values
# Some professions occur a lot more than others.
# Note: It looks like some manual cleaning should be done, but let's move on
X['field'].value_counts().plot(kind='barh', figsize=(5,40));


# In[13]:


# Distributions of numeric data
# Some are normally distributed, others are very skewed.
X.hist(layout=(20,4), figsize=(20,50));


# In[14]:


# What do people find important? Is this related to the outcome (match / no match)?
subset = ['attractive_important','ambition_important','attractive_partner','ambition_partner']
X_sub=X[subset].copy(deep=True) # Deep copy is overkill but avoids some warnings
X_sub['match'] = [classes[int(x)] for x in y] 
sns.set(style="ticks")
sns.pairplot(X_sub, hue="match");


# ## Part 1: Preprocessing

# ### Question 1.1: Simple pipeline (3 points)
# Implement a function `simple_pipeline` that returns an sklearn pipeline that preprocesses the data in a minimal way before running a classifier:
# - Categorical features:
#     - Impute missing values by replacing them with the most frequent value for that feature
#     - Perform one-hot encoding. Use `sparse=False` to avoid that it returns a sparse datasets. Use `handle_unknown='ignore'` to ignore categorical values that where not seen during training.
# - Numeric features:
#     - Remove missing values by replace missing values with the mean value for that feature
#     
# **Note:** this function should not internally use any variables that are not passed as arguments (e.g. categorical columns must be derived from `X`, not by using the global `categorical` variable).

# In[6]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Implement
def simple_pipeline(X, clf):
    """ Returns a minimal pipeline that imputes missing values and does one-hot-encoding for categorical features
    Keyword arguments:
    X -- The training data
    clf -- any scikit-learn classifier
    Returns: a scikit-learn pipeline which preprocesses the data and then runs the classifier
    """
    # The list of the names of all categorical features
    cat_cols = X.select_dtypes(include=["category"]).columns.tolist()

    cat_pipe = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(sparse=False, handle_unknown='ignore')
    )
    num_pipe = make_pipeline(
        SimpleImputer(strategy='mean')
    )
    
    preprocessor = make_column_transformer((cat_pipe, cat_cols), remainder=num_pipe)
    
    return make_pipeline(preprocessor, clf)


# #### Sanity check
# To be correct, this pipeline should be able to fit any classifier without error. Run this code to do a sanity check.

# In[21]:


from sklearn.tree import DecisionTreeClassifier
# simple_pipeline(X, DecisionTreeClassifier()).fit(X,y)


# ### Question 1.2: Feature count (1 point)
# How many features are being constructed by this pipeline (i.e., on how many features is the classifier trained)?

# In[22]:


# from sklearn.tree import DecisionTreeClassifier
# pipe = simple_pipeline(X, DecisionTreeClassifier()).fit(X,y)
# pipe['columntransformer'].fit(X)
# pd.DataFrame(pipe['columntransformer'].transform(X))


# In[23]:


# Fill in the correct answer, should be an integer. Don't change the name of the variable
q_1_2 = 287


# ### Question 1.3: Scaling (3 points)
# First, implement a function `scale_feature` which performs scaling on a given numeric feature, and returns the scaled results.

# In[7]:


from sklearn.preprocessing import StandardScaler

def scale_feature(X, scaler, feature):
    """ Scale numerical feature 'feature' of data 'X' with scaler 'scaler'
    
    X -- training data
    scaler -- a scikit-learn transformer (e.g. StandardScaler)
    feature -- the feature of which to return the transformed results
    
    returns:
     a list or 1D numpy array with the scaled values
    """
    feat = np.array(X[feature]).reshape(-1, 1)
    scaler.fit(feat)
    return scaler.transform(feat).ravel()


# Second, implement a function `plot_1_3` that plots the scaled values of the 'age'
# feature, scaled with three different scalers: power transform, min-max scaling,
# and standard scaling. You can use the default settings of the sklearn scalers. 
# 
# For the plotting, use the `plot_value_counts` function below, which uses your `scale_feature` function.
# 
# **Note:** Getting a plot output does not guarantee that your function implementation is correct. Interpret whether the values make sense.

# In[17]:


from collections import Counter

def plot_value_counts(X, scalers, feature):
    """ Plots the counts of scaled feature values of the given feature using the given scalers
    It will simply ignore all missing values.
      
    X -- training data
    scalers -- an array of sklearn scaling transformers
    feature -- the name of the feature to be scaled and plotted. Must be a string
    """
    for scaler in scalers:
        scaled_values = scale_feature(X, scaler, feature)
        non_nan = [v for v in scaled_values if not np.isnan(v)]
        value_counts = Counter(non_nan)
        ax = sns.scatterplot(x=value_counts.keys(), y=value_counts.values(), label=scaler.__class__.__name__)
    ax.set_xlabel("scaled value")
    ax.set_ylabel("count")
    ax.set_title("number of occurrences of each value after scaling")


# In[26]:


from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler

# Implement
def plot_1_3(X):
    """ Plots the scaled values of the 'age' feature, scaled with three different 
    scalers: power transform, min-max scaling, and standard scaling.
    """
    scalers = [
        PowerTransformer(),
        MinMaxScaler(),
        StandardScaler()
    ]
    plot_value_counts(X, scalers, 'age')
    
# plot_1_3(X)


# ### Question 1.4: Is scaling useful? (5 points)
# First, implement a function `flexible_pipeline`: an extension of your earlier pipeline which also allows you to add a feature scaling method for numeric features. The default is standard scaling. 'None' means no scaling should be applied.

# In[8]:


# Implement
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

def flexible_pipeline(X, clf, scaler=StandardScaler()):
    """ Returns a pipeline that imputes all missing values, encodes categorical features and scales numeric ones
    Keyword arguments:
    X -- The training data
    clf -- any scikit-learn classifier
    scaler -- any scikit-learn feature scaling method (Optional)
    Returns: a scikit-learn pipeline which preprocesses the data and then runs the classifier
    """
    # The list of the names of all numerical features
    num_cols = X.select_dtypes(exclude=["category"]).columns.tolist()
    
    pipe = simple_pipeline(X, clf)
    if scaler is not None:
        pipe['columntransformer'].remainder.steps.append(('scaler', scaler))
    
    return pipe


# In[19]:


from sklearn import set_config
set_config(display='diagram')

# Sanity check
# flexible_pipeline(X, DecisionTreeClassifier(), scaler=StandardScaler()).fit(X,y)


# Second, implement a function `plot_1_4` which plots a heatmap comparing several combinations of scaling methods and classifiers:
# * As classifiers, the following algorithms in their default hyperparameters settings: 
#     * SVM with RBF kernel
#     * Random Forest
# * As options, the following feature scaling options in their default settings:
#     * No scaling
#     * Standard scaling
#     * PowerTransformer
# * In all cases, use OneHotEncoder with `sparse=False` and `handle_unknown='ignore'`
# 
# You should evaluate all pipelines using AUC (area under the ROC curve) with 3-fold cross-validation. 
# Compare all methods with the same cross-validation folds, shuffle the data and use `random_state=1`.
# Where possible, also use `random_state=1` for the classifiers. 
# Only report the test scores (not the training scores).

# In[9]:


### Helper plotting function. Do not change.
def heatmap(columns, rows, scores):
    """ Simple heatmap.
    Keyword arguments:
    columns -- list of options in the columns
    rows -- list of options in the rows
    scores -- numpy array of shape (#rows, #columns) of scores
    """
    df = pd.DataFrame(scores, index=rows, columns=columns)
    sns.heatmap(df, cmap='Greens', linewidths=0.5, annot=True, fmt=".3f")


# In[81]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Implement
def plot_1_4(X, y):
    """ Evaluates 2 classifiers together with 3 types of scaling. See description above.
    """
    clfs = [
        SVC(kernel='rbf', random_state=1),
        RandomForestClassifier(random_state=1)
    ]
    scalers = [
        None,
        StandardScaler(),
        PowerTransformer()
    ]
    
    scores = np.zeros((len(clfs), len(scalers)))
    for clf_ix in range(len(clfs)):
        for scaler_ix in range(len(scalers)):
            pipe = flexible_pipeline(X, clfs[clf_ix], scaler=scalers[scaler_ix])
            kfold = StratifiedKFold(n_splits=3, random_state=1, shuffle=True)
            scores[clf_ix, scaler_ix] = np.mean(cross_val_score(pipe, X, y, cv=kfold, scoring='roc_auc'))
    
    clf_names = [clf.__class__.__name__ for clf in clfs]
    scaler_names = [scaler.__class__.__name__ for scaler in scalers]
    heatmap(scaler_names, clf_names, scores)

# plot_1_4(X, y)


# ### Question 1.5: Interpretation (2 point)
# Interpret the heatmap of Question 1.4. Which of the following are correct?  
# Enter your answer as a comma-separated string without spaces, e.g. "A,B,C"
# - 'A': Scaling or not scaling makes no difference for any of the classifiers.
# - 'B': Scaling is very important for both classifiers.
# - 'C': Scaling is important for SVMs, but not needed for Random Forests.
# - 'D': The default SVM with the right scaling technique performs slightly better than the default Random Forest.
# - 'E': The power transformer is much better than other techniques on this dataset because many features have a power law distribution (performance difference is more than 1%).
# - 'F': The power transformer about equally good as the standard scaler (performance difference is less than 1%)
# - 'G': No answer

# In[ ]:


# Fill in the correct answers, e.g. 'A,B,C'. Don't change the name of the variable
q_1_5 = 'C,D,F'


# ### Question 1.6: Grid search (2 points)
# 
# Perform a grid search for the SVM classifier. Range `C` on a log scale from 0.1 to 1000 (5 values), and `gamma` from 1e-4 to 1 (5 values).
# Scale the data using standard scaling and use one-hot-encoding before passing the data to the SVM, and make sure that there is no data leakage. 
# I.e. the test data must never be given to the preprocessings steps.
# 
# You should evaluate all pipelines using AUC (area under the ROC curve) with 3-fold cross-validation (more folds would be better, this is to save time). 
# Compare all methods with the same cross-validation folds, shuffle the data and use `random_state=1`.
# Where possible, also use `random_state=1` for the classifiers. 
# Only report the test scores (not the training scores).
# 
# Implement a function `plot_1_6` which plots a heatmap comparing the scores for all values of `C` and `gamma`. You can use the same heatmap function as before.

# In[49]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Implement
def plot_1_6(X, y):
    """ Plots a heatmap for a grid search varying the C and gamma values for a scaled SVM
    """
    C_space = np.geomspace(0.1, 1000, num=5)
    gamma_space = np.geomspace(1e-4, 1, num=5)
    param_grid = {
        'svc__C': C_space,
        'svc__gamma': gamma_space
    }
    
    clf = SVC(random_state=1)
    pipe = flexible_pipeline(X, clf, scaler=StandardScaler())
    kfold = StratifiedKFold(n_splits=3, random_state=1, shuffle=True)
    search = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=kfold, n_jobs=-1)
    search.fit(X, y)
    scores = np.reshape(search.cv_results_['mean_test_score'], (len(gamma_space), len(C_space)))
    
    C_labels = [f'$C = {C}$' for C in C_space]
    gamma_labels = [f'$\gamma = {gamma}$' for gamma in gamma_space]
    heatmap(C_labels, gamma_labels, scores)

# plot_1_6(X, y)


# ## Part 2: Feature importance
# In this part, we verify which features are most important to include in our models. In addition, we will check whether it would be interesting to build different models, with different sets of features, for different types of people.
# 
# 
# 
# 
# 

# ### Question 2.1: Specialized models (2 points)
# Implement a function `fit_by_feature` that does the following: 
# * Filter the input data by field of occupation (e.g. 'law' or 'engineering').
# * Fit your `flexible_pipeline`, without feature scaling and with a RandomForest classifier (1000 trees, 10% for max_features, random state 0) on the filtered input data.
# * Return the trained model
# 
# 

# In[10]:


from sklearn.ensemble import RandomForestClassifier

# Implement
def fit_by_feature(X, y, field=''):
    """ Returns a random forest model trained on data filtered by field of occupation. See detailed description above.
    Keyword arguments:
    X, y -- The input data
    field -- The field of occupation to filter on, e.g. 'engineering' or 'law'. Must be a string. 
    """
    if field != '':
        X_fil = X[X.field == field]
        y_fil = y[X.field == field]
    else:
        X_fil = X
        y_fil = y
    
    clf = RandomForestClassifier(n_estimators=1000, max_features=0.10, random_state=0, n_jobs=-1)
    pipe = flexible_pipeline(X_fil, clf, scaler=None)
    pipe.fit(X_fil, y_fil)
    return pipe


# In[119]:


# Sanity check
#fit_by_feature(X, y, field='engineering')


# ### Question 2.2: Cross-validation (3 points)
# Implement a function 'evaluate_by_feature' that evaluates the model trained by 
# 'fit_by_feature' using 5-fold stratified cross-validation and AUC. Then, use it to evaluate 3 models:
# - The model trained with all the data
# - The model trained for participants from the field "engineering"
# - The model trained for participants from the field "law"
# 
# For all models, use your `fit_by_feature` function. For the cross-validation, 
# use shuffling and random state 0.
# 
# Note: You can't use `cross_val_score` here, you need to implement a simple
# cross-validation loop around 'fit_by_feature' yourself.  
# Note 2: Technically, since the same people occur multiple times in this dataset,
# grouped cross-validation should be used. However, since the ID of these people
# is not in the dataset, there is no way to do this and you can ignore this.

# In[134]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score

def evaluate_by_feature(X, y, field='', fitter=fit_by_feature):
    """ Returns the mean AUC score of the given model based on stratified 5-fold cross-validation.
    Keyword arguments:
    X, y -- The input data
    field -- The field of occupation to filter on, e.g. 'engineering' or 'law'. Must be a string. 
    fitter -- The model to be evaluated. Must adhere to the interface of the fit_by_feature(X, y, field) function.
    """
    scorer = make_scorer(roc_auc_score, needs_threshold=True)
    scores = []
    kfold = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for train_index, test_index in kfold.split(X, y):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]
        clf = fitter(X_train, y_train, field)
        scores.append(scorer(clf, X_test, y_test))
    
    return np.mean(scores)

print(evaluate_by_feature(X, y))
print(evaluate_by_feature(X, y, field='law'))
print(evaluate_by_feature(X, y, field='engineering'))


# Interpret the AUC scores. Which of the following are correct?  
# Enter your answer as a comma-separated string without spaces-valued, e.g. "A,B,C"
# 
# *   'A': The models all have very similar performance (less that 5% difference)
# *   'B': It is harder to predict a good match for engineers than for lawyers, based on only the data available for that field.
# *   'C': It is harder to predict a good match for lawyers than for engineers, based on only the data available for that field.
# *   'D': Based on these results, it's not worth training per-field models, it's best to use all available data in one model.
# *   'E': No answer

# In[135]:


# Fill in the correct answers, e.g. 'A,B,C'. Don't change the name of the variable
q_2_2 = 'B,D'


# ### Question 2.3: Feature importance (3 points)
# Implement a function `plot_2_3` that does the following:  
# * Call your `fit_by_feature` function to train a pipeline on the filtered input data with fields "engineering", "law" and "film".
# * Retrieve the feature importances from the trained random forest and match them to the correct names. Depending on how you implemented your `flexible_pipeline` these are likely the first or the last columns in the processed dataset.
# * Plot the model-based feature importances using the plotting function `plot_importances` below.
# 
# *Notes*
# 
# * *Remember that the categorical features were encoded. Retrieve their encoded names from the one-hot-encoder (with `get_feature_names`).*

# In[136]:


# Plotting function. Do not edit.
def plot_importances(importance, feature_names, top_k=20, title=""):
    """ Simple bar chart plottinh feature importances
    Keyword arguments:
    importance -- The feature importances 
    feature_names -- The original features names in their original order
    top_k -- How many features to plot
    """
    # Trees
    sorted_idx = importance.argsort()[-top_k:]
    fig, ax = plt.subplots(figsize=(5, 5))
    y_ticks = np.arange(0, top_k)
    ax.barh(y_ticks, importance[sorted_idx])
    ax.set_yticklabels(feature_names[sorted_idx])
    ax.set_yticks(y_ticks)
    ax.set_title(title)

    fig.tight_layout()
    plt.show()


# In[239]:


# Implement
def plot_2_3(X, y, field=''):
    """ Returns a plot of the most important features, where data is filtered by field. See detailed description above.
    """
    cat_cols = X.select_dtypes(include=["category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["category"]).columns.tolist()
    
    pipe = fit_by_feature(X, y, field)
    enc = pipe['columntransformer'].named_transformers_['pipeline']['onehotencoder']
    cat_feats = enc.get_feature_names(cat_cols)
    # kind of guessing that the categorical features are the first columns
    feature_names = np.append(cat_feats, num_cols)
    importances = pipe['randomforestclassifier'].feature_importances_
    
    plot_importances(importances, feature_names, title=field)

# plot_2_3(X, y, field='engineering')
# plot_2_3(X, y, field='law')
# plot_2_3(X, y, field='film')


# ### Question 2.4: Interpretation (2 points)
# 
# Interpret the previous results of Question 2. Which of the following are correct?  
# Enter your answer as a comma-separated string without spaces, e.g. "A,B,C"
# - 'A': The matches for engineers, lawyers, and people in the film industry are decided by the exact same features.
# - 'B': Matches for engineers are strongly influenced by age, while this is much less the case for lawyers and people in the film industry.
# - 'C': Most matches depend strongly on whether the person was considered funny (funny_o) and the partner was considered funny (funny_partner), irrespective of profession.
# - 'D': There are significant differences in the feature importances based on field, but this does not translate into better per-field models.
# - 'E': No answer

# In[ ]:


# Fill in the correct answers, e.g. 'A,B,C'. Don't change the name of the variable
q_2_4 = 'B,C,D'


# ## Part 3: Threshold calibration

# ### Question 3.1: Tuning random forests (2 points)
# Use a grid search to optimize the RandomForest pipeline from question 2.1. Vary the number of trees from 100 to 1500 (at least 4 values) and set `max_features` to either
# 0.05, 0.1, or 'auto' (sqrt(p)). Evaluate all pipelines using AUC (area under the ROC curve) with 3-fold cross-validation. Compare all methods with the same cross-validation folds, shuffle the data and use `random_state=1`.
# Plot the results in a heatmap in function `plot_3_1`.

# In[148]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

#Implement
def plot_3_1(X, y):
    """ See detailed description above.
    """
    n_space = np.linspace(100, 1500, num=5, dtype=int)
    mf_space = [0.05, 0.1, 'auto']
    param_grid = {
        'randomforestclassifier__n_estimators': n_space,
        'randomforestclassifier__max_features': mf_space
    }
    
    clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    pipe = flexible_pipeline(X, clf, scaler=None)

    kfold = StratifiedKFold(3, shuffle=True, random_state=1)
    search = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=kfold, n_jobs=-1)
    search.fit(X, y)
    
    scores = np.reshape(search.cv_results_['mean_test_score'], (len(mf_space), len(n_space)))
    n_labels = [f'n = {n}' for n in n_space]
    mf_labels = [f'mf = {mf}' for mf in mf_space]
    heatmap(n_labels, mf_labels, scores)
    
# plot_3_1(X, y)


# ### Question 3.2: ROC curve (3 points)
# Implement a function `plot_3_2` that plots the ROC curve for the Random Forest pipeline with `n_estimators=1000`.
# Also indicate the point on the curve that corresponds to the 0.5 probability decision threshold and mentioned it on the plot legend.

# In[28]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

#Implement
def plot_3_2(X, y):
    """ See detailed description above.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

    clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=1)
    pipe = flexible_pipeline(X_train, clf, scaler=None)
    pipe.fit(X_train, y_train)
    
    fpr, tpr, thresholds = roc_curve(y_test, pipe.predict_proba(X_test)[:,1])

    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR (recall)")
    close_default = np.argmin(np.abs(thresholds - 0.5))
    plt.plot(fpr[close_default], tpr[close_default], '^', markersize=10,
             label="threshold 0.5", fillstyle="none", c='k', mew=2)
    plt.legend(loc=4)
    plt.show()
    
# plot_3_2(X, y)


# ### Question 3.3: Threshold calibration (3 points)
# Calibrate your model to get a higher recall. What would be the optimal decision threshold (approximately) assuming that a false negative (missing a good match) is twice as bad as a false positive (going on a date with someone who is not a good match)? Be careful that the dataset is imbalanced, so this must be taken into account. The goal is to find the threshold with the lowest actual cost (taking the usual ratio of positive and negative labels into account). Your results may vary slightly. The grade will depend on the distance to the actual optimum (within a tolerance).

# In[30]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

def cost(fpr, tpr, cost_FN, cost_FP, ratio_P):
    return fpr * cost_FP * ratio_P + (1 - tpr) * (1 - ratio_P) * cost_FN

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=1)
pipe = flexible_pipeline(X_train, clf, scaler=None)
pipe.fit(X_train, y_train)

fpr, tpr, thresholds = roc_curve(y_test, pipe.predict_proba(X_test)[:,1])

c_FN = 2
c_FP = 1
ratio_P = len(y[y == 1]) / len(y)

costs = [cost(fpr[x], tpr[x], c_FN, c_FP, ratio_P) for x in range(len(thresholds))]

min_cost = np.min(costs)
min_thres = np.argmin(costs)
min_thres_val = thresholds[np.argmin(costs)]

print(min_thres_val, fpr[min_thres], tpr[min_thres], min_cost)


# In[22]:


# Fill in the correct answer, should be a float. Don't change the name of the variable
q_3_3 = 0.012


# ## Part 4: Algoritmic bias
# Check whether there is any algorithmic bias in the predictions.

# ### Question 4.1 Racial bias (3 points)
# * Use the same RandomForest pipeline with 1000 trees and `max_features=0.1`
# * Use 3-fold stratified cross validation and return the predictions for every fold
# * Separate the test set predictions into different groups depending on the feature 'race', and report the ROC AUC for each group
# * Implement a function `plot_4_1` which returns a visualization of these results (e.g. a bar chart)

# In[134]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

#Implement
def plot_4_1(X, y):
    """ Returns a bar chart of the AUC measured, grouped by the value for the 'race' feature
    """
    clf = RandomForestClassifier(n_estimators=1000, max_features=0.1, n_jobs=-1, random_state=1)
    pipe = flexible_pipeline(X, clf, scaler=None)

    kfold = StratifiedKFold(3, shuffle=True, random_state=1)
    y_pred = cross_val_predict(pipe, X, y, cv=kfold, n_jobs=-1)
    
    scores = {}
    for race in X.race.unique():
        # Deal with NaN values
        if type(race) is float and np.isnan(race):
            y_true = y[X.race.isna()]
            y_score = y_pred[X.race.isna()]
            race = 'Missing'
        else:
            y_true = y[X.race == race]
            y_score = y_pred[X.race == race]
        
        scores[race] = roc_auc_score(y_true, y_score)
        
    plt.bar(scores.keys(), scores.values())
    plt.xticks(rotation='vertical')
    plt.show()
    
# plot_4_1(X, y)


# ### Question 4.2 (3 points)
# * Implement a function `plot_4_2` where you weight the samples (instances) with a weight inversely proportional to their frequency and then visualise the results in the same way as in question 4.1 (as a bar chart).
# * Interpret the results and explain them in `answer_q_4_2` (max 400 characters).

# In[135]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

#Implement
def plot_4_2(X, y):
    """ Returns a bar chart of the AUC measured, grouped by the value for the 'race' feature
    """
    clf = RandomForestClassifier(n_estimators=1000, max_features=0.1, n_jobs=-1, random_state=1)
    pipe = flexible_pipeline(X, clf, scaler=None)
    
    # Compute frequency of each race value
    race_freq = {}
    for race in X.race.unique():
        # Deal with NaN values
        if type(race) is float and np.isnan(race):
            y_race = y[X.race.isna()]
            race = 'Missing'
        else:
            y_race = y[X.race == race]

        race_freq[race] = len(y_race) / len(y)
        
    weights = []
    for instance_race in X.race:
        if type(instance_race) is float and np.isnan(instance_race):
            weights.append(1 / race_freq['Missing'])
        else:
            weights.append(1 / race_freq[instance_race])
            
    assert(len(weights) == len(X))

    kfold = StratifiedKFold(3, shuffle=True, random_state=1)
    y_pred = cross_val_predict(pipe, X, y, cv=kfold, n_jobs=-1, fit_params={
        'randomforestclassifier__sample_weight': weights
    })
    
    # Get ROC AUC scores per race
    scores = {}
    for race in X.race.unique():
        if type(race) is float and np.isnan(race):
            y_true = y[X.race.isna()]
            y_score = y_pred[X.race.isna()]
            race = 'Missing'
        else:
            y_true = y[X.race == race]
            y_score = y_pred[X.race == race]
        
        scores[race] = roc_auc_score(y_true, y_score)
    
    # Plot scores
    plt.bar(scores.keys(), scores.values())
    plt.xticks(rotation='vertical')
    plt.show()
    
# plot_4_2(X, y)


# In[147]:


answer_q_4_2 = """
We see that the unweighted model performs better for missing race values and worse for Asian race 
values. By assigning instance weights, we try to ensure that each race has equal influence on the model's
predictions, despite the fact that the data is imbalanced. Indeed, the weighted model shows less
variation in ROCAUC per race. Remaining variation could be due to chance.
               """
print("Answer is {} characters long".format(len(answer_q_4_2)))


# In[ ]:


print(time.time()-start_time)


# In[ ]:




last_edit = 'March 15, 2021'