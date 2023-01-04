# Classification-of-Comments-based-on-Toxicity

# Table of Content
1 Overview of the model
  Abstract
  Introduction
  Literature Review
2 Libraries required for the project
3 Data Explanation Analysis
  Loading of Data
  Analysis of Data
4 Feature Engineering

5 Modeling and Evaluation

  Hyperparameters Tuning

  Naive Bayes
  Logistic Regression
  Random Forest
  SVM
  
6 Ensembling
  Boosting
  Voting
  
7 Interpretation of the Observations
8 Results and conclusion

# Overview of the model
Abstract

Due to the threat of abuse and harassment, many people avoid from posting their opinions online. Social media has become a part of our lives and internet usage has 
dramatically increased in recent years. This has led to an increase in the amount of people sharing their opinions through social media comments. Since social media is
a public platform, it has become more prone to toxic comments. Platforms are currently having difficulty conducting discussions effectively, which has prompted several
communities to limit or prohibit user comments. The dataset we use consists of the comments posted by Wikipedia talk page editors. These comments have been classified 
as poisonous by human raters due to their wording. The several types of toxicity include threat, insult, filthy language, extreme toxicity, and identity hatred.

Introduction

Sometimes, cyberbullying techniques like threats, insults, and other behaviors find a home on social media platforms. Online social networks have a very large user base
. Consequently, it is crucial to safeguard network users against rudeness. Automatically identifying poisonous remarks is one of the main challenges of such activity. 
Text messages containing profane, racist, or otherwise harmful language are considered toxic. Toxicology is among the several publicly accessible models they have created 
so far and supplied through the Perspective API. The present models, however, continue to have flaws and do not permit users to pick the sorts of toxicity they are search
ing for (e.g., some platforms may be fine with profanity, but not with other types of toxic content). To determine the toxicity of online comments, we are using Natural
Language Processing to identify a solution. The field of ‘artificial intelligence’ known as ‘natural language processing’ is more specifically focused on giving computer
s human-like comprehension of spoken and written words. Numerous methodologies are used for human-free poisonous comment identification. The following statistics-based
criteria are frequently used: the length of the comment, the number of capital letters, exclamation points, question marks, spelling mistakes, the number of tokens that
contain non-alphabet symbols, the number of abusive, aggressive, and threatening words in the comment, etc. A comment has a higher risk of being labeled poisonous if it
contains a lot of foul language. Typographical and spelling errors might result in some words that are not often used. Often, those who leave harmful comments purposefully
twist their words.

Literature review :

Comments that are nasty, disrespectful, or have a tendency to have users leave the debate are referred to as toxic comments. We might conduct safer debates on 
different social networks, news websites, or online forums if these harmful comments could be automatically discovered. The manual moderation of comments is expensive,
inefficient, and occasionally impossible. Different machine learning techniques, namely various deep neural network designs, are used to automatically or semi-automatically
detect hazardous comments. The challenge of classifying hazardous comments has received a lot of attention recently, however there hasn't been a systematic literature 
analysis of this topic yet, making it challenging to determine its maturity, trends, and research gaps. Our main goal in this work was to address this by methodically 
listing, contrasting, and categorizing the prior research on the classification of poisonous comments in order to identify viable research avenues. The outcomes of this
comprehensive assessment of the literature are useful for academics and professionals involved in natural language processing. The results of the systematic review are
listed in the next section, along with a commentary on the findings. The final portion offers our conclusions and suggestions for additional research.


# Analysis of Data
 The training data comprises 159,571 observations in 8 columns compared to the test data's 153,164 observations in 2 columns.
 The graph below shows a frequency plot of comment length. The majority of comments are brief, with only a few exceeding 1000 words.
 3. It is clear that the label toxic has more results in the training dataset than threat, which has the fewest.
 The frequency plot for the labeled data is shown below. Because the majority of the remarks are regarded as non-toxic, a considerable class disparity is.
 Below are examples of a non-toxic comment and a toxic comment with the label 'toxic’ to better grasp what the comments look like.
 Identifying the labels that are most likely to appear with a comment could be useful.
 The cross-correlation matrix demonstrates that there is a strong likelihood that offensive remarks will also be offensive.
 We develop a tool to produce word clouds in order to gain insight into the words that are most important to certain labels. A label for a parameter is provided 
 to the function ( toxic, insult, threat, etc)
 
# Feature-engineering
  We must tokenize the comments in order to divide the statement down into distinct terms before fitting models. Punctuation and other special characters are eliminated
  during the tokenize() method. After analyzing the outcomes of feature engineering, we additionally removed non-ascii characters. After lemmatizing the comments, we 
  remove any that are less than three words in length.
  
# Benchmarking Different Vectorizer
Our research has led us to the concluding that TFIDF could be used to reduce weight of tokens that appear frequently in a corpus and, as a result, are statistically 
less insightful than attributes that appear only infrequently in the training corpus.

TFIDF was used in conjunction with CountVectorizer. However, it falls well short of TFIDF's performance. In fact, CountVectorizer comes before TfidfTransformer as the 
TfidfVectorizer. TfidfTransformer is a function that converts a count matrix to a standardized matrix. Tokens that appear frequently in a given corpus and are thus 
experimentally less insightful than features that appear in a small proportion of the training data are downscaled by using tf-idf rather than the raw frequencies of 
incidence of a token in a document collection. As a result, we can improve the precision in this situation.

Words like "wiki," "Wikipedia," "edit," and "page" are frequently used in this corpus since it contains data from revisions to the Wikipedia talk page, for instance. 
However, they don't give us any helpful information for our categorization objectives, which should likely be the purpose TFIDF performed better than CountVectorizer. 
Following the transition, we may examine some of the properties listed.

# Modeling and Evaluation
Baseline Model
Our baseline model is Naive Bayes, more precisely Multinomial Naive Bayes.

We also wish to compare various models, especially ones that are effective in classifying text. So, instead of comparing Logistic Regression and Linear Support Vector 
Machine, we decided to compare Multinomial Naive Bayes.

Evaluation Metrics
Since we have 6 labels, the average of those 6 labels will serve as the F1-score, which we use as our main metric for assessing the performance of our models. We will 
also consider additional metrics, like as Hamming loss and recall, when evaluating models.


Cross Validation
Cross Validation is how we compare the baseline model to the other two models we've selected ‘LogisticRegression and LinearSVC’.
In general, the linear SVC model and the logistic regression model outperform other models, as shown by the cross validation discussed above. Due to the fact that the 
threat label and identity hate label have the fewest observations, Multinomial Naive Bayes does not perform well as a baseline model. Now that we have the test dataset,
we want to examine how well these three models do with actual predictions.

# Evaluation and Modeling
The performance of the several models following their training on the test data is compared in the results table and plot above. In general, Linear SVC excels the other
models according to the F1 score, but Muninomial Naive Bayes underperforms the other two models.

# Confusion Matrix visualization
The label toxic's confusion matrix is displayed below. Due to the majority of our data being non-toxic, it is important to note that all models fairly accurately forecast 
Non-toxic labels. Linear SVC is quite effective at categorizing toxic comments; however Multinomial NB tends to forecast more toxic to non-toxic comments.

Based on the aforementioned analysis, we can conclude that LinearSVC outperforms everyone for the "hazardous" label for these three models with default settings.

# Results and Conclusions
The Linear SVC performs the best in terms of the evaluation metric. However, we think we will achieve better outcomes after fine-tuning the hyperparameters for assembly. The fastest 
train model is linear SVC. In terms of interpretability, Linear SVC also has a simpler internal processing and is simpler for people to understand. As a result, we settle on Linear SVC as the best model.
