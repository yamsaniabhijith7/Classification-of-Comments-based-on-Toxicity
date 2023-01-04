#!/usr/bin/env python
# coding: utf-8

# # A model for categorizing toxic content in user comments

# <a id='table-of-content'></a>
# ## Table of Content
# 
# 
# 1. [Overview of the model](#Overview)
#  - [Abstract](#ab)
#  - [Introduction](#in)
#  - [Literature Review](#li)
#  
# 
# 2. [Libraries required for the project](#Libraries)
# 2. [Data Explanation Analysis](#EDA)
#  - [Loading of Data](#Data-loading)
#  - [Analysis of Data](#Data-Analysis)
# 
# 3. [Feature Engineering](#Feature-engineering)
# 4. [Modeling and Evaluation](#Modeling)
#  
# 5. [Hyperparameters Tuning](#Tuning)
#  - [Naive Bayes](#nb-tuning)
#  - [Logistic Regression](#lr-tuning)
#  - [Random Forest](#rf-tuning)
#  - [SVM](#svm-tuning)
#  
# 6. [Ensembling](#Ensembling)
#  - [Boosting](#Boosting)
#  - [Voting](#Voting)
# 7. [Interpretation of the Observations](#Interpretation)
# 8. [Results and conclusion](#result)
# 

# <a id='overview'></a>
# ## Overview of the model
# 
# <a id="ab"> </a>
# Abstract
# 
# Due to the threat of abuse and harassment, many people avoid from posting their opinions online. Social media has become a part of our lives and internet usage has dramatically increased in recent years. This has led to an increase in the amount of people sharing their opinions through social media comments. Since social media is a public platform, it has become more prone to toxic comments.  Platforms are currently having difficulty conducting discussions effectively, which has prompted several communities to limit or prohibit user comments. The dataset we use consists of the comments posted by Wikipedia talk page editors. These comments have been classified as poisonous by human raters due to their wording. The several types of toxicity include threat, insult, filthy language, extreme toxicity, and identity hatred.
# 
# <a id="in"> </a>
# Introduction
# 
# Sometimes, cyberbullying techniques like threats, insults, and other behaviors find a home on social media platforms. Online social networks have a very large user base. Consequently, it is crucial to safeguard network users against rudeness. Automatically identifying poisonous remarks is one of the main challenges of such activity. Text messages containing profane, racist, or otherwise harmful language are considered toxic. Toxicology is among the several publicly accessible models they have created so far and supplied through the Perspective API. The present models, however, continue to have flaws and do not permit users to pick the sorts of toxicity they are searching for (e.g., some platforms may be fine with profanity, but not with other types of toxic content). To determine the toxicity of online comments, we are using Natural Language Processing to identify a solution. The field of ‘artificial intelligence’ known as ‘natural language processing’ is more specifically focused on giving computers human-like comprehension of spoken and written words. Numerous methodologies are used for human-free poisonous comment identification. The following statistics-based criteria are frequently used: the length of the comment, the number of capital letters, exclamation points, question marks, spelling mistakes, the number of tokens that contain non-alphabet symbols, the number of abusive, aggressive, and threatening words in the comment, etc. A comment has a higher risk of being labeled poisonous if it contains a lot of foul language. Typographical and spelling errors might result in some words that are not often used. Often, those who leave harmful comments purposefully twist their words.
# 
# <a id="Li"> </a>
# Literature review : 
# 
# Comments that are nasty, disrespectful, or have a tendency to have users leave the debate are referred to as toxic comments. We might conduct safer debates on different social networks, news websites, or online forums if these harmful comments could be automatically discovered. The manual moderation of comments is expensive, inefficient, and occasionally impossible. Different machine learning techniques, namely various deep neural network designs, are used to automatically or semi-automatically detect hazardous comments. The challenge of classifying hazardous comments has received a lot of attention recently, however there hasn't been a systematic literature analysis of this topic yet, making it challenging to determine its maturity, trends, and research gaps. Our main goal in this work was to address this by methodically listing, contrasting, and categorizing the prior research on the classification of poisonous comments in order to identify viable research avenues. The outcomes of this comprehensive assessment of the literature are useful for academics and professionals involved in natural language processing. The results of the systematic review are listed in the next section, along with a commentary on the findings. The final portion offers our conclusions and suggestions for additional research.
# 
# 

# <a id='Libraries requied for the project'></a>
# ## Libraries requied for the project

# In[8]:


import string
import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from timeit import default_timer as timer


# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import  roc_auc_score,precision_score,accuracy_score,recall_score, roc_curve,f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix
from statistics import mean
from sklearn.metrics import hamming_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve


# In[10]:


from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import recall_score
import statistics

from collections import Counter
from wordcloud import WordCloud

from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

import warnings
import xgboost as xgb
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='EDA'></a>
# ## Data Explanation Analysis

# <a id='Loading of Data'></a>
# ### Loading of Data

# In[12]:


train_dt = pd.read_csv("./train.csv")
test_dt = pd.read_csv("./test.csv")
test_dt_y = pd.read_csv("./test_labels.csv")


# ### Analysis of Data
# 
# 1.      The training data comprises 159,571 observations in 8 columns compared to the test data's 153,164 observations in 2 columns.
# 2.      The graph below shows a frequency plot of comment length. The majority of comments are brief, with only a few exceeding 1000 words.
# 3.      3. It is clear that the label toxic has more results in the training dataset than threat, which has the fewest.
# 4.      The frequency plot for the labeled data is shown below. Because the majority of the remarks are regarded as non-toxic, a considerable class disparity is.
# 5.      Below are examples of a non-toxic comment and a toxic comment with the label 'toxic’ to better grasp what the comments look like.
# 6.      Identifying the labels that are most likely to appear with a comment could be useful.
# 7.      The cross-correlation matrix demonstrates that there is a strong likelihood that offensive remarks will also be offensive.
# 8.      We develop a tool to produce word clouds in order to gain insight into the words that are most important to certain labels. A label for a parameter is provided to the function ( toxic, insult, threat, etc)
# 9.      Select a class to see the terms that contribute to that class the most: insult.

# In[13]:


train_dt.head()


# In[14]:


train_dt.describe()


# In[15]:


test_dt.head()


# In[16]:


test_dt_y.head()


# Notice that the training data contains 159,571 observations with 8 columns and the test data contains 153,164 observations with 2 columns.

# In[17]:


train_dt.shape


# In[18]:


test_dt.shape


# Below is a plot showing the comment length frequency. As noticed, most of the comments are short with only a few comments longer than 1000 words.

# In[19]:


sns.set(color_codes=True)
comnt_length = train_dt.comment_text.str.len()
sns.distplot(comnt_length, kde=False, bins=20, color="blue")


# Further exploratory shows that label `toxic` has the most observations in the training dataset while `threat` has the least.

# In[20]:


# Labels from training data.
train_lbl = train_dt[['toxic','obscene','insult','severe_toxic','identity_hate','threat']]
label_ct = train_lbl.sum()


# In[21]:


label_ct.plot(kind='bar', title='Labels Frequency', color='green')


# Below is the plot for the labeled data frequency. There is significant class imbalance since majority of the comments are considered non-toxic.

# In[22]:


# Code for creating a bar graph to show the distribution of classes within each label.
barsize = 0.50

bars_1 = [sum(train_dt['toxic'] == 1), sum(train_dt['obscene'] == 1), sum(train_dt['insult'] == 1), sum(train_dt['severe_toxic'] == 1),
         sum(train_dt['identity_hate'] == 1), sum(train_dt['threat'] == 1)]
bars_2 = [sum(train_dt['toxic'] == 0), sum(train_dt['obscene'] == 0), sum(train_dt['insult'] == 0), sum(train_dt['severe_toxic'] == 0),
         sum(train_dt['identity_hate'] == 0), sum(train_dt['threat'] == 0)]

r1 = np.arange(len(bars_1))
r2 = [x + barsize for x in r1]

plt.bar(r1, bars_1, color='yellow', width=barsize, label='labeled = 1')
plt.bar(r2, bars_2, color='green', width=barsize, label='labeled = 0')

plt.xlabel('group', fontweight='bold')
plt.xticks([r + barsize for r in range(len(bars_1))], ['toxic','obscene','insult','severe_toxic','identity_hate','threat'])
plt.legend()
plt.show()


# To get a better understanding of what the comments look like, below are examples of one clean (non-toxic) comment and one toxic (specifically, with label "toxic") comment.

# In[23]:


#clean comment example
train_dt.comment_text[1]


# In[24]:


# toxic comment example
train_dt[train_dt.toxic == 1].iloc[1, 1]


# It could be a good practice to see which labels are likely to appear alongside a comment.

# In[25]:


# Cross-labeled correlation matrix
row_sums = train_dt.iloc[:, 2:].sum(axis=1)
tem = train_dt.iloc[:, 2:-1]
train_cor = tem[row_sums > 0]
corr = train_cor.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True, cmap="Blues")


# Obscene comments are likely to be insulting, as evidenced by the cross-correlation matrix.

# We write a function to create word clouds to get an idea of which words contribute the most to various labels. The function accepts a parameter label (i.e., toxic, insult, threat, etc)

# In[28]:


def W_Cld(token):
    """
    Visualize the most common words contributing to the token.
    """
    threat_context = train_dt[train_dt[token] == 1]
    threat_text = threat_context.comment_text
    neg_text = pd.Series(threat_text).str.cat(sep=' ')
    wordcloud = WordCloud(width=1600, height=800,
                          max_font_size=200).generate(neg_text)

    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud.recolor(colormap="Blues"), interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Most common words assosiated with {token} comment", size=20)
    plt.show()


# In[29]:


# Enter the label name into the interactive text box from insult,toxic,threat
tkn = input(
    'Choose a class to represent the most common terms that belong to that class. :')
W_Cld(tkn.lower())


# <a id='Feature-engineering'></a>
# ## Feature-engineering

# We must tokenize the comments in order to divide the statement down into distinct terms before fitting models. Punctuation and other special characters are eliminated during the tokenize() method. After analyzing the outcomes of feature engineering, we additionally removed non-ascii characters. After lemmatizing the comments, we remove any that are less than three words in length. 

# In[30]:


test_lbl = ["toxic", "severe_toxic", "obscene",
               "threat", "insult", "identity_hate"]


# In[31]:


def tokenization(txt):
    '''
    Tokenize text and return a non-unique list of tokenized words found in the text. 
    Normalize to lowercase, strip punctuation, remove stop words, filter non-ascii characters.
    Lemmatize the words and lastly drop words of length < 3.
    '''
    txt = txt.lower()
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunc = regex.sub(" ", txt)
    wrds = nopunc.split(' ')
    # remove any non ascii
    wrds = [wrd.encode('ascii', 'ignore').decode('ascii') for wrd in wrds]
    lmtzr = WordNetLemmatizer()
    wrds = [lmtzr.lemmatize(w) for w in wrds]
    wrds = [w for w in wrds if len(w) > 2]
    return wrds


# #### Benchmarking Different Vectorizer

# Our research has led us to the concluding that TFIDF could be used to reduce weight of tokens that appear frequently in a corpus and, as a result, are statistically less insightful than attributes that appear only infrequently in the training corpus.
# 
# TFIDF was used in conjunction with CountVectorizer. However, it falls well short of TFIDF's performance. In fact, CountVectorizer comes before TfidfTransformer as the TfidfVectorizer. TfidfTransformer is a function that converts a count matrix to a standardized matrix. Tokens that appear frequently in a given corpus and are thus experimentally less insightful than features that appear in a small proportion of the training data are downscaled by using tf-idf rather than the raw frequencies of incidence of a token in a document collection. As a result, we can improve the precision in this situation.
#  
# Words like "wiki," "Wikipedia," "edit," and "page" are frequently used in this corpus since it contains data from revisions to the Wikipedia talk page, for instance. However, they don't give us any helpful information for our categorization objectives, which should likely be the purpose TFIDF performed better than CountVectorizer.
# 2.  	Following the transition, we may examine some of the properties listed below.

# In[32]:


import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[35]:


vctr = TfidfVectorizer(ngram_range=(1, 1), analyzer='word',
                         tokenizer=tokenization, stop_words='english',
                         strip_accents='unicode', use_idf=1, min_df=10)
X_train_dt = vctr.fit_transform(train_dt['comment_text'])
X_test_dt = vctr.transform(test_dt['comment_text'])


# Following the transformation, we can look at some of the features listed below.

# In[36]:


vctr.get_feature_names()[0:20]


# <a id='Modeling'></a>
# ## Modeling and Evaluation

# ### Baseline Model

# Our baseline model is Naive Bayes, more precisely Multinomial Naive Bayes.
#  
# We also wish to compare various models, especially ones that are effective in classifying text. So, instead of comparing Logistic Regression and Linear Support Vector Machine, we decided to compare Multinomial Naive Bayes.
# 

# ### Evaluation Metrics

# Since we have 6 labels, the average of those 6 labels will serve as the F1-score, which we use as our main metric for assessing the performance of our models. We will also consider additional metrics, like as Hamming loss and recall, when evaluating models.

# <a id='cv'></a>
# ### Cross Validation

# 1.  	Cross Validation is how we compare the baseline model to the other two models we've selected ‘LogisticRegression and LinearSVC’.
# 2.  	In general, the linear SVC model and the logistic regression model outperform other models, as shown by the cross validation discussed above. Due to the fact that the threat label and identity hate label have the fewest observations, Multinomial Naive Bayes does not perform well as a baseline model. Now that we have the test dataset, we want to examine how well these three models do with actual predictions.

# In[37]:


#Initially creating classification model with default parameters.
clsf1 = LinearSVC()
clsf2 = LogisticRegression()
clsf3 = MultinomialNB()


# In[38]:


def crs_valdtn_score(classifier, X_train, y_train):
    '''
    Iterate though each label and return the cross validation F1 and Recall score 
    '''
    mthd = []
    nme = classifier.__class__.__name__.split('.')[-1]

    for label in test_lbl:
        recall = cross_val_score(
            classifier, X_train, y_train[label], cv=10, scoring='recall')
        f1 = cross_val_score(classifier, X_train,
                             y_train[label], cv=10, scoring='f1')
        mthd.append([nme, label, recall.mean(), f1.mean()])

    return mthd


# In[42]:


# Calculating the F1 and Recall results for our three baseline models using cross validation.
mthd1_cv = pd.DataFrame(crs_valdtn_score(clsf1, X_train_dt, train_dt))
mthd2_cv = pd.DataFrame(crs_valdtn_score(clsf2, X_train_dt, train_dt))
mthd3_cv = pd.DataFrame(crs_valdtn_score(clsf3, X_train_dt, train_dt))


# In[43]:


# Creating a dataframe to show summary of results.
mthd_cv = pd.concat([mthd1_cv, mthd2_cv, mthd3_cv])
mthd_cv.columns = ['Model', 'Label', 'Recall', 'F1']
meth_cv = mthd_cv.reset_index()
meth_cv[['Model', 'Label', 'Recall', 'F1']]


# Based on the cross validation results, we discovered that the linear SVC model and the Logistic Regression model perform much better overall. Multinomial Naive Bayes does not work well as a baseline model, particularly for the 'threat' and 'identity hate' labels, which have the fewest observations.
# 
# Now we'll look at how these three models fare on the exact prediction - the test dataset.

# ### Evaluation and Modeling
# 
# The performance of the several models following their training on the test data is compared in the results table and plot above. In general, Linear SVC excels the other models according to the F1 score, but Muninomial Naive Bayes underperforms the other two models.

# In[44]:


def score(classifier, X_train_dt, y_train, X_test_dt, y_test):
    """
    Calculate Hamming-loss, F1, Recall for each label on test dataset.
    """
    methods = []
    hloss = []
    name = classifier.__class__.__name__.split('.')[-1]
    predict_df = pd.DataFrame()
    predict_df['id'] = test_dt_y['id']

    for label in test_lbl:
        classifier.fit(X_train_dt, y_train[label])
        predicted = classifier.predict(X_test_dt)

        predict_df[label] = predicted

        recall = recall_score(y_test[y_test[label] != -1][label],
                              predicted[y_test[label] != -1],
                              average="weighted")
        f1 = f1_score(y_test[y_test[label] != -1][label],
                      predicted[y_test[label] != -1],
                      average="weighted")

        conf_mat = confusion_matrix(y_test[y_test[label] != -1][label],
                                    predicted[y_test[label] != -1])

        methods.append([name, label, recall, f1, conf_mat])

    hamming_loss_score = hamming_loss(test_dt_y[test_dt_y['toxic'] != -1].iloc[:, 1:7],
                                      predict_df[test_dt_y['toxic'] != -1].iloc[:, 1:7])
    hloss.append([name, hamming_loss_score])

    return hloss, methods


# In[45]:


# Calculating the Hamming-loss F1 and Recall score for our 3 baseline models.
h1, mthd1 = score(clsf1, X_train_dt, train_dt, X_test_dt, test_dt_y)
h2, mthd2 = score(clsf2, X_train_dt, train_dt, X_test_dt, test_dt_y)
h3, mthd3 = score(clsf3, X_train_dt, train_dt, X_test_dt, test_dt_y)


# In[46]:


# Creating a dataframe to show summary of results.
mthd1 = pd.DataFrame(mthd1)
mthd2 = pd.DataFrame(mthd2)
mthd3 = pd.DataFrame(mthd3)
mthd = pd.concat([mthd1, mthd2, mthd3])
mthd.columns = ['Model', 'Label', 'Recall', 'F1', 'Confusion_Matrix']
meth = mthd.reset_index()
meth[['Model', 'Label', 'Recall', 'F1']]


# In[47]:


# Visualizing F1 score results through box-plot.
axs = sns.boxplot(x='Model', y='F1', data=mthd, palette="Blues")
sns.stripplot(x='Model', y='F1', data=mthd,
              size=8, jitter=True, edgecolor="green", linewidth=2, palette="Blues")
axs.set_xticklabels(axs.get_xticklabels(), rotation=20)

plt.show()


# The results table and plot above show a comparison of these different models after training and how these models perform on test data.
# 
# It is worth noting that Muninomial Naive Bayes does not outperform the other two models, whereas Linear SVC outperforms them all in terms of F1 score.

# ### Visualzing performance till now for each classifier across each category

# In[48]:


# Code to create bar graph of F1 and Recall across each label for Multinomial Naive Bayes
print("Naive Bayes Multinomial regression plotting")
m2 = mthd[mthd.Model == 'MultinomialNB']

m2.set_index(["Label"], inplace=True)
get_ipython().run_line_magic('matplotlib', 'inline')
m2.plot(figsize=(16, 8), kind='bar', title='Metrics',
        rot=60, ylim=(0.0, 1), colormap='tab10')


# In[49]:


# Code to create bar graph of F1 and Recall across each label for Logistic regression
print("Logistic regression plotting")
m2 = mthd[mthd.Model == 'LogisticRegression']

m2.set_index(["Label"], inplace=True)
get_ipython().run_line_magic('matplotlib', 'inline')
m2.plot(figsize=(16, 8), kind='bar', title='Metrics',
        rot=60, ylim=(0.0, 1), colormap='tab10')


# In[50]:


# Linear SVC code to generate a bar chart of F1 as well as recall over each label
print("Plot for Linear SVC")
m2 = mthd[mthd.Model == 'LinearSVC']

m2.set_index(["Label"], inplace=True)
get_ipython().run_line_magic('matplotlib', 'inline')
m2.plot(figsize=(16, 8), kind='bar', title='Metrics',
        rot=60, ylim=(0.0, 1), colormap='tab10')


# <a id='model-comparison'></a>
# ###  Confusion Matrix visualization
# 
# 1.  	The label toxic's confusion matrix is displayed below. Due to the majority of our data being non-toxic, it is important to note that all models fairly accurately forecast Non-toxic labels. Linear SVC is quite effective at categorizing toxic comments; however Multinomial NB tends to forecast more toxic to non-toxic comments.
#  
# 2.  	Based on the aforementioned analysis, we can conclude that LinearSVC outperforms everyone for the "hazardous" label for these three models with default settings.

# In[51]:


def drawConfusionMatrix(cm):
    """
    Plot Confusion matrix of input cm.
    """
    cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    ax = plt.axes()
    sns.heatmap(cm,
                annot=True,
                annot_kws={"size": 16},
                cmap="Blues",
                fmt='.2f',
                linewidths=2,
                linecolor='steelblue',
                xticklabels=("Non-toxic", "Toxic"),
                yticklabels=("Non-toxic", "Toxic"))

    plt.ylabel('True', fontsize=18)
    plt.xlabel('Predicted', fontsize=18)
    plt.show()


# In[52]:


def Matrix(label):
    """
    Plot Confusion matrix for each label and call function drawConfusionMatrix().
    """
    print(f"*************** {label} labelling ***************")
    labels = {"toxic": 0, "severe_toxic": 1, "obscene": 2,
              "threat": 3, "insult": 4, "identity_hate": 5}

    pos = labels[label]
    for i in range(pos, len(meth), 6):
        print()
        print(f"****  {meth['Model'][i]}  ***")
        cm = meth['Confusion_Matrix'][i]
        drawConfusionMatrix(cm)


# The confusion matrix for the label 'toxic' is shown below. Because the majority of our data is non-toxic, all model suggests Non-toxic labels fairly well. However, Multinomial NB tries to predict more hateful comments to non-toxic comments, whereas Linear SVC excels at classifying toxic comments.

# In[53]:


token = input('Choose a class for the Confusion Matrix: ')
Matrix(token.lower())


# Based on the comparison above, we can conclude that for these 3 models with default options, **LinearSVC outperforms everyone for the 'toxic' label. **
# 
# 
# 

# ### Aggregated Hamming Loss Score
# Since it has the fewest inaccurate labels among all models, Logistic Regression is performing exceptionally well.
# 

# In[54]:


# Creating a dataframe to summarize Hamming-loss
hls1_df = pd.DataFrame(h1)
hls2_df = pd.DataFrame(h2)
hls3_df = pd.DataFrame(h3)


# In[55]:


hamgloss = pd.concat([hls1_df, hls2_df, hls3_df])
hamgloss.columns = ['Model', 'Hamming_Loss']
hls = hamgloss.reset_index()
hls[['Model', 'Hamming_Loss']]


# **Logistic Regression** outperforms all other models because it has the lowest rate of incorrect labels.

# <a id='pipeline'></a>
# ### Pipelines
# 1.  	Only model comparisons without any hyperparameter modification have been done so far. Let's utilize pipeline to clean up the code, then apply a few hand-picked hyperparameters to see how each model performs. We choose to manually tweak class weight for the models to see if we can get better results because the imbalanced data is now the biggest worry.
#  
# We will concentrate on these two models as they perform better: linear SVM and logistic regression. Only the average F1 score, Recall, and Hamming Loss will be displayed for the sake of comparison.
#  
# 2.  	Take note of how better the results are than the simple models after modifying class weight. By about 1%, linear SVC performs better than logistic regression.

# In[56]:


pipe_lr = Pipeline([
    ('lr', LogisticRegression(class_weight="balanced"))
])

pipe_linear_svm = Pipeline([
    ('svm', LinearSVC(class_weight={1: 20}))
])

pipelines = [pipe_lr, pipe_linear_svm]


# In[57]:


score_df = []
for pipe in pipelines:
    f1_values = []
    recall_values = []
    hl = []
    training_time = []
    predict_df = pd.DataFrame()
    predict_df['id'] = test_dt_y['id']
    for label in test_lbl:
        start = timer()
        pipe.fit(X_train_dt, train_dt[label])
        train_time = timer() - start
        predicted = pipe.predict(X_test_dt)
        predict_df[label] = predicted

        f1_values.append(f1_score(
            test_dt_y[test_dt_y[label] != -1][label], predicted[test_dt_y[label] != -1], average="weighted"))
        recall_values.append(recall_score(
            test_dt_y[test_dt_y[label] != -1][label], predicted[test_dt_y[label] != -1], average="weighted"))
        training_time.append(train_time)
        name = pipe.steps[-1][1].__class__.__name__.split('.')[-1]

    hamming_loss_score = hamming_loss(
        test_dt_y[test_dt_y['toxic'] != -1].iloc[:, 1:7], predict_df[test_dt_y['toxic'] != -1].iloc[:, 1:7])

    val = [name, mean(f1_values), mean(recall_values),
           hamming_loss_score, mean(training_time)]
    score_df.append(val)


# In[58]:


scores = pd.DataFrame(score_df,)
scores.columns = ['Model', 'F1', 'Recall', 'Hamming_Loss', 'Training_Time']
scores


# We can see that after adjusting 'class weight,' we get much better results than that of the basic models. By about 1%, LinearSVC outperforms LogisticRegression.

# <a id='Tuning'></a>
# ## Hyperparameter Tuning with Grid Search 

# Now we decide to do grid search to seek for the "optimal" hyperparameters for the basic models that we've chose. Later we will make comparison based on the best model from each algorithm, since we have 6 different lables, tuning models for each label would be time expensive, so we will use the most common label "Toxic" to tune hyperparameters.

# <a id='lr-tuning'></a>
# ### Logistic Regression Tuning

# In[59]:


logtc_regression_clsfr = LogisticRegression()

parameter_grid = {'solver': ['newton-cg', 'lbfgs', 'liblinear'],
                  'class_weight': [None, 'balanced']}

crs_valdtn = StratifiedKFold(n_splits=5)

gridsearch = GridSearchCV(logtc_regression_clsfr,
                           param_grid=parameter_grid,
                           cv=crs_valdtn,
                           scoring='f1')

gridsearch.fit(X_train_dt, train_dt['toxic'])

print('Best parameters: {}'.format(gridsearch.best_params_))

gridsearch.best_estimator_


# <a id='svm-tuning'></a>
# ### SVM Classifier Tuning

# In[60]:


svm_clsfr = LinearSVC()

parameter_grid = {'class_weight': [None, 'balanced'],
                  'C': [1, 5, 10]}

crs_valdtn = StratifiedKFold(n_splits=5)

grid_search = GridSearchCV(svm_clsfr,
                           param_grid=parameter_grid,
                           cv=crs_valdtn,
                           scoring='f1')

gridsearch.fit(X_train_dt, train_dt['toxic'])

print('Best parameters: {}'.format(gridsearch.best_params_))

gridsearch.best_estimator_


# ###  Model Selection

# We will then compare these two models based on their tunned hyperparameters, we will also include training time as one of the metric when we compare models.

# In[61]:


svm_clf = LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
                    intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                    multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                    verbose=0)
lr_clf = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                     intercept_scaling=1, max_iter=100, multi_class='ovr',
                                     n_jobs=None, penalty='l2', random_state=None, solver='lbfgs',
                                     tol=0.0001, verbose=0, warm_start=False)

tunnd_mdl_score_df = []
for model in [svm_clf, lr_clf]:
    f1_values = []
    recall_values = []
    hl = []
    training_time = []
    predict_df = pd.DataFrame()
    predict_df['id'] = test_dt_y['id']

    for label in test_lbl:
        start = timer()
        model.fit(X_train_dt, train_dt[label])
        training_time.append(timer() - start)
        predicted = model.predict(X_test_dt)
        predict_df[label] = predicted

        f1_values.append(f1_score(test_dt_y[test_dt_y[label] != -1][label],
                                  predicted[test_dt_y[label] != -1],
                                  average="weighted"))
        recall_values.append(recall_score(test_dt_y[test_dt_y[label] != -1][label],
                                          predicted[test_dt_y[label] != -1],
                                          average="weighted"))
        name = model.__class__.__name__

    hamming_loss_score = hamming_loss(test_dt_y[test_dt_y['toxic'] != -1].iloc[:, 1:7],
                                      predict_df[test_dt_y['toxic'] != -1].iloc[:, 1:7])

    val = [name, mean(f1_values), mean(recall_values),
           hamming_loss_score, sum(training_time)]

    tunnd_mdl_score_df.append(val)


# In[62]:


tunnd_scores = pd.DataFrame(tunnd_mdl_score_df,)
tunnd_scores.columns = ['Model', 'F1',
                       'Recall', 'Hamming_Loss', 'Traing_Time']
tunnd_scores


# <a id='Ensembling'></a>
# ## Ensembling 

# We want to see if ensembling could help us achieve better results because it improves machine learning outcomes by incorporating several models and allows for greater predictive performance than a single model.
# 
# To ensemble different models, we firstly tried some models based on tree boosting, then use a voting classfier to ensemble one of the boosting model with the basic models in previous parts.

# <a id='boosting'></a>
# ### Boosting Models

# We tested three popular tree-based boosting models and compared them.

# In[63]:


ab_clsf = AdaBoostClassifier()
gb_clsf = GradientBoostingClassifier()
xgb_clsf = xgb.XGBClassifier()
bstng_mdls = [ab_clsf, gb_clsf, xgb_clsf]


# In[64]:


bstng_score_df = []
for mdl in bstng_mdls:
    f1_values = []
    recall_values = []
    training_time = []
    hloss = []
    predict_df = pd.DataFrame()
    predict_df['id'] = test_dt_y['id']

    for idx, label in enumerate(test_lbl):
        strt = timer()
        model.fit(X_train_dt, train_dt[label])
        predicted = model.predict(X_test_dt)
        training_time.append(timer() - start)
        predict_df[label] = predicted
        f1_values.append(f1_score(test_dt_y[test_dt_y[label] != -1][label],
                                  predicted[test_dt_y[label] != -1],
                                  average="weighted"))
        recall_values.append(recall_score(test_dt_y[test_dt_y[label] != -1][label],
                                          predicted[test_dt_y[label] != -1],
                                          average="weighted"))
        name = model.__class__.__name__

    hamming_loss_score = hamming_loss(test_dt_y[test_dt_y['toxic'] != -1].iloc[:, 1:7],
                                      predict_df[test_dt_y['toxic'] != -1].iloc[:, 1:7])

    val = [name, mean(f1_values), mean(recall_values),
           hamming_loss_score, mean(training_time)]

    bstng_score_df.append(val)


# ### Scores After Boosting the Model

# In[65]:


bstng_score = pd.DataFrame(bstng_score_df,)
bstng_score.columns = ['Model', 'F1',
                          'Recall', 'Hamming_Loss', 'Training_Time']
bstng_score


# We decide to use gradient boosting because it outperforms the other two boosting models.

# <a id='voting'></a>
# ### VotingClassifier

# In[66]:


ensemble_clf = VotingClassifier(estimators=[('lr', lr_clf),
                                            ('svm', svm_clf),
                                            ('xgb', xgb_clsf)], voting='hard')
ensemble_score_df = []
f1_values = []
recall_values = []
hl = []
training_time = []

predict_df = pd.DataFrame()
predict_df['id'] = test_dt_y['id']
for label in test_lbl:
    start = timer()
    ensemble_clf.fit(X_train_dt, train_dt[label])
    training_time.append(timer() - start)
    predicted = ensemble_clf.predict(X_test_dt)
    predict_df[label] = predicted
    f1_values.append(f1_score(test_dt_y[test_dt_y[label] != -1][label],
                              predicted[test_dt_y[label] != -1],
                              average="weighted"))
    recall_values.append(recall_score(test_dt_y[test_dt_y[label] != -1][label],
                                      predicted[test_dt_y[label] != -1],
                                      average="weighted"))
    name = 'Ensemble'

hamming_loss_score = hamming_loss(test_dt_y[test_dt_y['toxic'] != -1].iloc[:, 1:7],
                                  predict_df[test_dt_y['toxic'] != -1].iloc[:, 1:7])

val = [name, mean(f1_values), mean(recall_values),
       hamming_loss_score, mean(training_time)]
ensemble_score_df.append(val)


# printing the values
ensemble_score = pd.DataFrame(ensemble_score_df,)
ensemble_score.columns = ['Model', 'F1',
                          'Recall', 'Hamming_Loss', 'Training_Time']
ensemble_score


# Note that while the ensembled model performed admirably, it could not outperform LinearSVC because the ensemled model's hyperparameters were not tuned.

# ## Interpretation of the Observations

# Examining the words that Logistic Classifier misclassified. Investigating the 'toxic' label
# 

# In[67]:


lbl = 'toxic'
lr = LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
               intercept_scaling=1, loss='squared_hinge', max_iter=1000,
               multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
               verbose=0)
lr.fit(X_train_dt, train_dt[label])
Toxic_LR = lr.predict(X_test_dt)
test_combined = pd.concat([test_dt, test_dt_y], axis=1)


# In[68]:


commentCheck = test_combined[(test_combined.toxic == 1) & (
    Toxic_LR == 0)].comment_text
commentCheck.shape


# - 1347 were mislabeled as non-toxic when they were actually toxic.

# In[69]:


# extract wrongly classified comments
commentCheck = test_combined[(test_combined.toxic == 1) & (
    Toxic_LR == 0)].comment_text

neg_Check = pd.Series(commentCheck).str.cat(sep=' ')
wordcloud = WordCloud(width=1600, height=800,
                      max_font_size=200).generate(neg_Check)
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud.recolor(colormap="Blues"), interpolation='bilinear')
plt.axis("off")
plt.title("Most often used misclassified words", size=20)
plt.show()


# - We want to analyze why the model couldn't recognize these words. Were they not present in the training set? 
# - In order to analyze, we first need to pass these raw comment strings through same tokenizer and check the common tokens.

# In[70]:


import nltk
nltk.download('stopwords')


# In[71]:


from nltk.corpus import stopwords


# In[72]:


wrongWords = tokenization(neg_Check)
stoplist = set(stopwords.words("english"))
wrongWords = [w for w in wrongWords if w not in stoplist]
ctr = Counter(wrongWords)
ctr.most_common(20)


# 'ucking' is a frequent word in the test set, and it appears that our classifier has yet to learn to classify it as toxic. Let's see how common this word was in the training data.

# In[73]:


neg_text_train = train_dt['comment_text'].str.cat(sep=' ')
ctr_train = Counter(tokenization(neg_text_train))
ctr_train.get('ucking')


# It's worth noting that this token was uncommon in our training set. That explains why our model was unable to learn it.
# It also gives us some ideas for how we can improve our model further.

# Let's look at the features to see if this word has a high feature importance.

# ### 👓 Visual check how logistic learns

# In[74]:


def plot_lrng_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Plot learning rate curve for the estimator with title, training data as X, 
    labels as y.
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator,
                                                            X, y, train_sizes=train_sizes, cv=cv, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="steelblue",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="olive",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[75]:


title = "Learning Curves for the TOXIC comments in linear support vector classification"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
                      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                      multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                      verbose=0)
plot_lrng_curve(estimator, title, X_train_dt,
                    train_dt['toxic'], ylim=(0.7, 1.01), cv=cv, n_jobs=4)


# <a id='result'> </a>
# ## Results and Conclusions

# The Linear SVC performs the best in terms of the evaluation metric. However, we think we will achieve better outcomes after fine-tuning the hyperparameters for assembly. The fastest train model is linear SVC. In terms of interpretability, Linear SVC also has a simpler internal processing and is simpler for people to understand. As a result, we settle on Linear SVC as the best model.
