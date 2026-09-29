
from datasets import Dataset


data = {
    "question": [
        # Question 1
        "What is the main objective of the study?",

        # Question 2
        "What does RFQ stand for, and what percentage of RFQs historically resulted in a sale?",

        # Question 3
        "Which machine learning model was selected as the final model?",

        # Question 4
        "Why is the prediction problem considered an imbalanced classification problem?",

        # Question 5
        "What techniques were investigated to deal with class imbalance?",

        # Question 6
        "Name three important features used by the final model.",

        # Question 7
        "Why could Quote features not be used as input features for the prediction model?",

        # Question 8
        "Why were RFQs with all customer and part source features missing removed from the dataset?",

        # Question 9
        "Why was the classification threshold changed from 0.5 to 0.62?",

        # Question 10
        "How does this approach differ from traditional spare-parts demand forecasting?"
    ],

    "answer": [
        # AI-generated answer for Question 1
        "The main objective of the study is to propose a method to use advance demand information, taking the form of request for quotations (RFQs), in a 'smart' manner. Specifically, it aims to estimate the likelihood of each individual RFQ being successful and forecast spare part demand based on unstructured data retrieved from these RFQs using machine learning and natural language processing techniques. ",

        # AI-generated answer for Question 2
        "RFQ stands for Request for Quotation. Historically, 17% of RFQs turn into a sale.",

        # AI-generated answer for Question 3
        "Based on the context provided, the Random Forest model was selected as the final model for predicting RFQ success probability. This is indicated by the statement: ‘We do this through visualization of a single random tree from the Random Forest (Fig. 11), upon which is zoomed-in on levels 1–2.’ The majority vote from the 400 trees in the Random Forest determines the prediction outcome.",

        # AI-generated answer for Question 4
        """The prediction problem is considered an imbalanced classification problem because the average ratio that an RFQ (Request for Quotation) ever becomes a sale in the historical dataset is only about 17%. This means that out of all the RFQs submitted, only about 17% of them eventually result in a sale. 
Additionally, within our specific dataset, 19% of quote lines converted into a sale. Given these high proportions (17% and 19%), it indicates an imbalance where the majority class (‘No Sale’) significantly outnumbers the minority class (‘Sale’). This imbalance makes it challenging for classification algorithms to accurately identify and predict the minority class instances (sales), as they are less frequent and often overlooked due to their rarity. 

Theoretical studies indicate that imbalanced data, characterized by unequal frequency distributions of instances among classes, can lead to significant drawbacks in performance when using standard classifier algorithms. These algorithms typically assume a balanced distribution of classes and equal misclassification costs, which do not apply well to imbalanced datasets like the one we have. Therefore, the prediction problem is considered an imbalanced classification problem due to this inherent imbalance between the majority class (‘No Sale’) and the minority class (‘Sale’).

""",

        # AI-generated answer for Question 5
        """In dealing with class imbalance, the following techniques were investigated in the research:

1. **Boosting**: Specifically, the Gradient Boosting Classifier algorithm was used.
2. **Cost-sensitive learning**: This indirectly addressed class imbalance by using the F1-score performance measure.
3. **Data resampling**:
   - Under-sampling: Excluding instances (usually majority class).
   - Over-sampling: Replicating/creating new instances (usually minority class).

The research preferred under-sampling over over-sampling due to practical considerations such as model training time, especially when incorporating cross-validation.
""",

        # AI-generated answer for Question 6
        """Three important features used by the final model are:

1. **Hit rate account**: The percentage of total quoted value for an account that converted to a sale at the time the quote was issued.
2. **Frequency part**: Frequency count of a part number in the sales order data at the time the quote was issued.
3. **Stock**: Whether the requested part was on stock, with its delivery window being ≤ 7 days.

These features are ranked as follows:

1. Hit rate account
2. Frequency part
3. Stock
""",

        # AI-generated answer for Question 7
        "Quote features cannot be used as input features for the prediction model because these are not available at the point in time an RFQ arrives. This means that when the quote is created, the information about whether a sales order will eventually result from it (hit rate account and hit rate part) has not yet been determined since the final outcome depends on subsequent actions such as creating a sales order. Therefore, using these features would introduce bias into the model, as they are based on outcomes that have not occurred at the time of the quote creation.",

        # AI-generated answer for Question 8
        "RFQs with all customer and part source features missing were removed from the dataset because these quotes did not contain any valid data for customers or parts that have been sold in the past. According to the context, when a quote contains a customer/part that has not been sold in the past, all feature values will be missing. This poses a potential data leak where there is no actual sale record available, which could lead to incorrect predictions and analysis if included in the dataset. To eliminate this issue, all rows for which all feature values of the customer- and/or part data source are missing were removed from the dataset.",

        # AI-generated answer for Question 9
        "The classification threshold was changed from 0.5 to 0.62 as a result of optimizing it based on validation curves, which showed that the optimal threshold lies within the range [0.55, 0.65]. After zooming in on this area and analyzing further data, the optimal classification threshold given the optimal feature subset and hyper-parameter configuration was determined to be 0.62.",

        # AI-generated answer for Question 10
        """This approach differs from traditional spare parts demand forecasting in several key ways:

1. **Focus on Success Probability**: Unlike traditional methods that focus on point forecasts of demand using historical sales data, this research estimates the likelihood of an RFQ (Request for Quotation) to be successful.

2. **Individual RFQ Focus**: The predictions are coupled with each individual piece of information (RFQ), focusing on the likelihood of each specific RFQ rather than a general forecast based on historical sales data.

3. **Use of Machine Learning and Natural Language Processing**: Instead of predominantly using time series models, this research uses machine learning and natural language processing to forecast spare part demand based on unstructured data retrieved from RFQs.

4. **Consideration as Advance Demand Information (ADI)**: The approach is closer to advance demand information (ADI) since RFQs can be considered ADI, which allows for a more nuanced understanding of future demand patterns.

5. **Case Study Approach**: Unlike other studies on ADI, this paper presents a case study to demonstrate how RFQs can be used as an ADI and how the resulting information is utilized in estimating the success probability that an RFQ will become a sale.

These differences highlight how this approach leverages modern AI techniques like machine learning and natural language processing to better understand and predict future demand patterns from unstructured data sources such as RFQs.

"""
    ],

    "contexts": [
        # Retrieved context for Question 1
        [
            """future demand information coming from customers, taking the form of 
requests for quotation (RFQs). RFQs are uncommitted requests for a quote 
of spare parts and/or exchange of parts, by means of email containing 
unstructured text, that do not necessarily result in a sale. Despite the fact 
that they are uncommitted, RFQs can be used to predict future demand 
using artificial intelligence techniques, e.g. supervised machine learning 
and natural language processing. 
The research in this paper is a case study carried out at a large after- 
sales service and maintenance provider, to which we will refer as the 
service provider. The service provider receives a large number of RFQs. 
Yet, the average ratio that an RFQ ever becomes a sale is only about 
17%. Furthermore, these large number of RFQs exceed capacity of em-
ployees responsible for responding to the RFQs. This increases the 
respond time of the service provider to an RFQ, which is an important

ployees responsible for responding to the RFQs. This increases the 
respond time of the service provider to an RFQ, which is an important 
factor for the success of a sale. Consequently, customers may complain 
and even move to a competitor. Therefore, it is important for the service 
provider to pick up RFQs that have higher chance of sale as not to waste 
the efforts of sales employees, which are expensive and scarce. Thus 
there is a clear need/opportunity for the service provider to process its 
RFQs in a ’smart’ manner. 
The objective of this research is two-fold. First, we propose a method 
to use advance demand information, taking the form of request for 
* Corresponding author. 
E-mail address: david.rohaan@hotmail.com (D. Rohaan).  
Contents lists available at ScienceDirect 
Expert Systems With Applications 
journal homepage: www.elsevier.com/locate/eswa 
https://doi.org/10.1016/j.eswa.2021.115925

Contents lists available at ScienceDirect 
Expert Systems With Applications 
journal homepage: www.elsevier.com/locate/eswa 
https://doi.org/10.1016/j.eswa.2021.115925  
Received 4 April 2021; Received in revised form 8 August 2021; Accepted 16 September 2021 
 
readable documents (Jiang, 2012). Two fundamental tasks of informa - 
tion extraction are named entity recognition (NER) and relation  
extraction. First, a named entity is a sequence of words that refers to a  
real-world entity. NER identifies named entities from unstructured text  
and classifies them into a set of predefined types. Usually, NER cannot be  
simply accomplished by string/pattern matching against pre-compiled  
dictionaries, so-called entity ruler, because (1) named entities usually  
do not form a closed set and (2) named entities can be context depen - 
dent. In such cases a NER model could provide a solution, which iden - 
tifies and categorizes entities in unstructured data, on the basis of  
unstructured labelled training data. Second, relation extraction is the  
task of detecting and characterizing the semantic relations between  
entities in text, which is less relevant for our research objective.  
2.4. Spare parts demand forecasting 
 
task of detecting and characterizing the semantic relations between  
entities in text, which is less relevant for our research objective.  
2.4. Spare parts demand forecasting  
There are several papers on forecasting spare parts demand. The  
main focus of this stream of research is to forecast slow moving erratic,  
lumpy, or intermittent demand patterns, which are typical characteris - 
tics of spare parts demand. One of the seminal works is Croston (1972).  
Several other papers propose approaches to extend this work e.g., Syn- 
tetos and Boylan (2005), Teunter, Syntetos, and Babai (2011) and Babai,  
Dallery, Boubaker, and Kalai (2019). We refer to Boylan and Syntetos  
(2010) and Pinçe, Turrini, and Meissner (2021) for reviews. Although  
our aim is to forecast spare parts demand also, we differ from this  
literature in three ways:   
1. Unlike these papers, who focus on point forecast of demand using  
demand historical sales data, we estimate the likelihood of an RFQ to  
be successful. 
 
literature in three ways:   
1. Unlike these papers, who focus on point forecast of demand using  
demand historical sales data, we estimate the likelihood of an RFQ to  
be successful.   
2. In this regard, we focus on the likelihood of each individual RFQ, and  
therefore, our predictions are coupled with each individual piece of  
information (RFQ).   
3. Therefore, our methodology is also different. Unlike those paper on  
forecasting spare parts, which predominantly use time series models,  
we use machine learning and natural language processing to forecast  
spare part demand based on unstructured data retrieved from RFQs.  
Furthermore, particularly because of (1) and (2), our paper is much  
closer to advance demand information (ADI) since RFQs can be  
considered ADI (Topan, Tan, van Houtum, & Dekker, 2018). Yet,  
different from those papers on ADI, we are presenting a case study to  
show how RFQs can be used as an ADI and how this information is used 
 
reasons, consequently making performance seem better than it actually  
is and complete loss of information due to the wrong choice of encoding.  
The remainder of the paper is structured as follows: Section 2 pre- 
sents the theoretical background and discusses our contribution to  
literature. Section 3 discusses our method. The results are presented in  
Section 4. In Section 5 we conclude our research and discuss the main  
findings of our work, indicate the limitations, and lay the foundation for  
further research. Finally, we discuss the ethical considerations of our  
research in Section 6.  
2. Literature review  
This section provides an overview of prior work relevant to achieving  
our research objective; B2B sales forecasting, imbalanced data classifi - 
cation, automation using NLP and spare part demand forecasting,  
including studies on advance demand information (ADI).  
2.1. Business-to-business sales potential forecasting 
 
representation based on historical sales data. This step requires identi - 
fying feature data describing a sales opportunity and deriving additional  
custom features, so called ‘meta variables’ (Mortensen, Christison, Li,  
Zhu, & Venkatesan, 2019). The second step is the data preparation,  
entailing data cleaning, -transformation and -splitting to prevent  
‘Garbage in, Garbage out’. The objective of the third step is to identify  
the model that is best in predicting sales potential. Here, we extended  
the existing methodology by incorporating a feature selection method - 
ology (Bohanec et al., 2015a), hyperparameter tuning and classification  
threshold optimization. The final step uses ML techniques and  
visualizations to gain/emphasize insights which the sales department  
can use to adjust their current mental decision models.  
Despite major progress within forecasting methods as a result of  
advancements in machine learning research, B2B sales forecasting 
 
model in a fully autonomous/automated flow, meaning that any un - 
recognized partial customer email address and/or part number by  
definition will never become a sale. Note that training a NER model  
(Section 2.3), which does account for context, was also attempted but  
had shown high losses during training indicating, and following from  
performance, that the NER model was unable to learn properly and thus  
unsuitable for the task at hand.  
4.3. Model insights  
In this section we want to give insight into the prediction model,  
which we expect will be perceived by most to be a ‘black-box’, with the  
objective to foster adoption in practice. First, the 13 most informative  
features which the prediction model uses to achieve its performance can  
be seen below in Table 6 with a description, ranked in descending order  
of importance. Table 6 shows that ABC encoded features (4 out of 13)  
and custom meta-features (3 out of 13) cover more than half of the most 
 
"""
        ],

        # Retrieved context for Question 2
        [
            """analytics applications, e.g. machine learning (ML). The majority, an 
estimated 80–90% of big data is unstructured data (e.g. emails), which is 
furthermore growing faster than any other type of data. Unstructured 
data is information that does not have a recognizable structure; it comes 
in many forms and thus is not a good fit for a mainstream database 
(Gandomi & Haider, 2015). A solution to analyzing such big unstruc -
tured data is natural language processing (NLP), which is a subfield of 
artificial intelligence that gives machines the ability to read, understand 
and derive meaning from human languages (Hirschberg & Manning, 
2015). Furthermore, NLP is said to have the ability to automate data 
extraction from large volumes of unstructured text (Li & Elliot, 2019). 
In this paper we focus on B2B sales forecasting using advance or 
future demand information coming from customers, taking the form of 
requests for quotation (RFQs). RFQs are uncommitted requests for a quote

future demand information coming from customers, taking the form of 
requests for quotation (RFQs). RFQs are uncommitted requests for a quote 
of spare parts and/or exchange of parts, by means of email containing 
unstructured text, that do not necessarily result in a sale. Despite the fact 
that they are uncommitted, RFQs can be used to predict future demand 
using artificial intelligence techniques, e.g. supervised machine learning 
and natural language processing. 
The research in this paper is a case study carried out at a large after- 
sales service and maintenance provider, to which we will refer as the 
service provider. The service provider receives a large number of RFQs. 
Yet, the average ratio that an RFQ ever becomes a sale is only about 
17%. Furthermore, these large number of RFQs exceed capacity of em-
ployees responsible for responding to the RFQs. This increases the 
respond time of the service provider to an RFQ, which is an important

ployees responsible for responding to the RFQs. This increases the 
respond time of the service provider to an RFQ, which is an important 
factor for the success of a sale. Consequently, customers may complain 
and even move to a competitor. Therefore, it is important for the service 
provider to pick up RFQs that have higher chance of sale as not to waste 
the efforts of sales employees, which are expensive and scarce. Thus 
there is a clear need/opportunity for the service provider to process its 
RFQs in a ’smart’ manner. 
The objective of this research is two-fold. First, we propose a method 
to use advance demand information, taking the form of request for 
* Corresponding author. 
E-mail address: david.rohaan@hotmail.com (D. Rohaan).  
Contents lists available at ScienceDirect 
Expert Systems With Applications 
journal homepage: www.elsevier.com/locate/eswa 
https://doi.org/10.1016/j.eswa.2021.115925

Expert
Systems With Applications 188 (2022) 115925
5
3.2.2. Handling missing values 
We determined the percentage missing values (MVs) per feature and 
addressed each feature based on the nature of their missingness. 
There exist three types of missing data: Missing At Random (MAR), 
Missing Completely At Random (MCAR) and Missing Not At Random 
(MNAR). First, MAR means that the distribution of MVs depends on the 
observed data and unknown parameter(s), but not on any missing data. 
Second, MCAR means that the distribution of MVs is independent of 
both the observed- and missing data and depends entirely on some un-
known parameter(s). Finally, MNAR means that the distribution of MVs 
is dependent on the MVs itself and therefore signifies meaning (García, 
Luengo, & Herrera, 2015). 
After verifying the nature of missingness for each feature with pro -
cess experts at the service provider, we differentiate between two ap -
proaches for handling MVs. First, instances with ≥ 1 MAR/MCAR

cess experts at the service provider, we differentiate between two ap -
proaches for handling MVs. First, instances with ≥ 1 MAR/MCAR 
feature values are discarded. We discard these instances because there is 
plenty of data and imputation, even a sophisticated method such as 
multiple imputation, would induce unnecessary uncertainty. Second, 
missing values of MNAR features are replaced by ‘unknown’. 
3.3. Data splitting 
Historically 17% of RFQs turn into a sale, meaning we are dealing 
with unequal distribution of instances amongst classes, i.e. class 
imbalance. When dealing with imbalanced data, it is best to take either a 
balanced- or stratified sample. Both types of samples force a certain 
composition to the to-be predicted target feature, where in a balanced 
sample this composition is predefined and in a stratified sample the 
composition follows the natural class distribution of the original data 
set. 
In our research, we use (random) stratified splitting, ensuring equal

composition follows the natural class distribution of the original data 
set. 
In our research, we use (random) stratified splitting, ensuring equal 
frequency distribution of classes in the data used for model training and 
-evaluation (Kuhn & Johnson, 2019). Moreover, we combine the strat -
ified splitting with k-fold cross validation to prevent overfitting. This 
means that data is split into k folds with equal class distribution, the 
model is trained on k-1 folds and evaluated on the remaining fold. This 
process is repeated for multiple splits, in which the training/testing fold 
allocation differ. The final performance estimation is the average per -
formance estimation of all k testing folds over all splits. 
3.4. Model training & building 
3.4.1. Outlier detection 
Outliers only exist in non-categorial features. To detect outliers, Z- 
scoring method is applied to the numerical features. This method as -

Expert
Systems With Applications 188 (2022) 115925
2
quotation (RFQ) data, to predict B2B sales. To achieve this goal, we use 
(i) supervised machine learning to predict likelihood of sale of RFQs and 
subsequently prioritize RFQs on sales potential, and (ii) natural lan -
guage processing to automate this prioritization by automatically 
extracting required input data from RFQs and feeding it into the pre -
diction model. Using our method, we enable the service provider to use 
its limited resources in the best way to generate maximum sales. Second, 
with our research we seek to bridge the gap between theory and practice 
by giving step-by-step guidance on incorporating supervised machine 
learning in B2B sales forecasting, revealing potential pitfalls along the 
way. Examples of such pitfalls are information leaking, due to various 
reasons, consequently making performance seem better than it actually 
is and complete loss of information due to the wrong choice of encoding.

literature in three ways:  
1. Unlike these papers, who focus on point forecast of demand using 
demand historical sales data, we estimate the likelihood of an RFQ to 
be successful.  
2. In this regard, we focus on the likelihood of each individual RFQ, and 
therefore, our predictions are coupled with each individual piece of 
information (RFQ).  
3. Therefore, our methodology is also different. Unlike those paper on 
forecasting spare parts, which predominantly use time series models, 
we use machine learning and natural language processing to forecast 
spare part demand based on unstructured data retrieved from RFQs. 
Furthermore, particularly because of (1) and (2), our paper is much 
closer to advance demand information (ADI) since RFQs can be 
considered ADI (Topan, Tan, van Houtum, & Dekker, 2018). Yet, 
different from those papers on ADI, we are presenting a case study to 
show how RFQs can be used as an ADI and how this information is used

formance by + 155.3% over prior manual handling and demonstrated 
the feasibility of automation through a proof of concept. The need for 
the sales forecasting model sprung, amongst other things, from the fact 
that the service provider receives more RFQs than it is able to process. 
Using the model in the future will allow the service provider to generate 
more sales, assuming the same distribution of ‘Sale’/’No sale’ amongst 
the RFQs that previously could not be processed, by responding to RFQs 
in order of descending predicted probability of sale. 
Regarding the limitations and future research direction, first, our 
prediction model predicts solely sales potential but neglects the value of 
the corresponding RFQ. This means that a low-cost part (e.g. bolt) could 
be prioritized over a high value part (e.g. engine) with neglectable dif-
ference in sales potential. Second, in our research we focused on opti -
mizing the F1 score, the harmonic mean between precision and recall.
"""
        ],

        # Retrieved context for Question 3
        [
            """Expert
Systems With Applications 188 (2022) 115925
11
(Witten, Frank, & Hall, 2011). Each association rule is represented with 
three standard evaluation metrics: support, confidence and lift. The 
support measures the proportion of cases in the data set which contain 
the antecedent of a rule. The confidence reflects the proportion of the 
cases in which the antecedent and consequent are satisfied. The lift re -
ports the ratio of the observed support to the expected antecedent and 
consequent being independent. Table 7 shows the top 25 association 
rules which were extracted from the training dataset, under the re -
strictions of ≥ 10% and ≥ 60% for minimal support and -confidence 
respectively. From Table 7 follows for example that selling non-rotable 
parts from stock to airlines (line 23) and selling low contributing non- 
rotable parts to defense (line 24) often result in a sale. 
Finally, we want to give insight into the prediction algorithm,

parts from stock to airlines (line 23) and selling low contributing non- 
rotable parts to defense (line 24) often result in a sale. 
Finally, we want to give insight into the prediction algorithm, 
Random Forest, itself. We do this through visualization of a single 
random tree from the Random Forest (Fig. 11), upon which is zoomed-in 
on levels 1–2. The model consists of 400 trees similar to the one shown in 
Fig. 11, from which the majority vote determines the prediction 
outcome. 
5. Conclusion 
In this section, we discuss the main findings of our work, indicate the 
limitations, and lay the foundation for further research. The research in 
our paper, a combination of supervised machine learning and natural 
language processing, has shown to increase B2B sales forecasting per -
formance by + 155.3% over prior manual handling and demonstrated 
the feasibility of automation through a proof of concept. The need for

formance by + 155.3% over prior manual handling and demonstrated 
the feasibility of automation through a proof of concept. The need for 
the sales forecasting model sprung, amongst other things, from the fact 
that the service provider receives more RFQs than it is able to process. 
Using the model in the future will allow the service provider to generate 
more sales, assuming the same distribution of ‘Sale’/’No sale’ amongst 
the RFQs that previously could not be processed, by responding to RFQs 
in order of descending predicted probability of sale. 
Regarding the limitations and future research direction, first, our 
prediction model predicts solely sales potential but neglects the value of 
the corresponding RFQ. This means that a low-cost part (e.g. bolt) could 
be prioritized over a high value part (e.g. engine) with neglectable dif-
ference in sales potential. Second, in our research we focused on opti -
mizing the F1 score, the harmonic mean between precision and recall.

literature in three ways:  
1. Unlike these papers, who focus on point forecast of demand using 
demand historical sales data, we estimate the likelihood of an RFQ to 
be successful.  
2. In this regard, we focus on the likelihood of each individual RFQ, and 
therefore, our predictions are coupled with each individual piece of 
information (RFQ).  
3. Therefore, our methodology is also different. Unlike those paper on 
forecasting spare parts, which predominantly use time series models, 
we use machine learning and natural language processing to forecast 
spare part demand based on unstructured data retrieved from RFQs. 
Furthermore, particularly because of (1) and (2), our paper is much 
closer to advance demand information (ADI) since RFQs can be 
considered ADI (Topan, Tan, van Houtum, & Dekker, 2018). Yet, 
different from those papers on ADI, we are presenting a case study to 
show how RFQs can be used as an ADI and how this information is used

different from those papers on ADI, we are presenting a case study to 
show how RFQs can be used as an ADI and how this information is used 
to estimate the success probability that an RFQ will become a sale. 
3. B2B sales potential forecasting model 
In this section we explain the methodology used to create the ma -
chine learning model step-by-step, structured according to the frame -
work of supervised machine learning (Fig. 2). 
Supervised machine learning (Fig. 2) starts by collecting historical 
labelled data (data whose class is known) of instances, reflecting fea -
tures (that could be) of importance in predicting class label. Then, data 
cleaning takes place after which the cleaned data is split into a training- 
and testing data set (usually ratio 70/30 or 80/20 respectively). Next, 
different classifiers are trained on different feature subsets of the 
training data and afterwards applied to the testing data. Accordingly, a

different classifiers are trained on different feature subsets of the 
training data and afterwards applied to the testing data. Accordingly, a 
performance estimation can be derived by comparing the predicted 
outcomes of the testing data with the true known outcomes of the testing 
data. Finally, the combination of feature subset and classifier that yields 
the best prediction performance is adopted and applied to future/unseen 
cases. 
3.1. Data acquisition 
3.1.1. From RFQ to sales order 
The process from RFQ to sales order at the service provider is shown 
in Fig. 3. If an RFQ is deemed to have the potential to result in a sale, a 
quote is created containing the offer (price, lead- time, etc.) in response 
to the RFQ. A quote may contain multiple (quote) lines, each repre -
senting the requested quantity for a specific part by the corresponding 
customer. Consequently, each quote line may or may not convert into a 
sales order.

representation based on historical sales data. This step requires identi -
fying feature data describing a sales opportunity and deriving additional 
custom features, so called ‘meta variables’ (Mortensen, Christison, Li, 
Zhu, & Venkatesan, 2019). The second step is the data preparation, 
entailing data cleaning, -transformation and -splitting to prevent 
‘Garbage in, Garbage out’. The objective of the third step is to identify 
the model that is best in predicting sales potential. Here, we extended 
the existing methodology by incorporating a feature selection method -
ology (Bohanec et al., 2015a), hyperparameter tuning and classification 
threshold optimization. The final step uses ML techniques and 
visualizations to gain/emphasize insights which the sales department 
can use to adjust their current mental decision models. 
Despite major progress within forecasting methods as a result of 
advancements in machine learning research, B2B sales forecasting

model in a fully autonomous/automated flow, meaning that any un -
recognized partial customer email address and/or part number by 
definition will never become a sale. Note that training a NER model 
(Section 2.3), which does account for context, was also attempted but 
had shown high losses during training indicating, and following from 
performance, that the NER model was unable to learn properly and thus 
unsuitable for the task at hand. 
4.3. Model insights 
In this section we want to give insight into the prediction model, 
which we expect will be perceived by most to be a ‘black-box’, with the 
objective to foster adoption in practice. First, the 13 most informative 
features which the prediction model uses to achieve its performance can 
be seen below in Table 6 with a description, ranked in descending order 
of importance. Table 6 shows that ABC encoded features (4 out of 13) 
and custom meta-features (3 out of 13) cover more than half of the most

mining, inference, and prediction. Springer.  
Heinze, G., & Dunkler, D. (2016). Five myths about variable selection. Transplant 
International, 30(1), 1–6. 
Hirschberg, Julia, & Manning, Christopher D. (2015). Advances in natural language 
processing. Science, 349(6245), 261–266. 
Japkowicz, N. (2001). Concept-learning in the presence of between-class and within- 
class imbalances. Canadian Conference on AI. Springer. 
Japkowicz, N., & Stephen, S. (2002). The class imbalance problem: A systematic study. 
Intelligent Data Analysis, 31. 
Jiang, J. (2012). Information extraction from text. In C. C. Aggarwal, & C. (. Zhai, Mining 
text data. Springer. 
Kuhn, M., & Johnson, K. (2019). Feature Engineering and Selection: A Practical Approach for 
Predictive Models ((1 ed.).). CRC Press.  
Lambert, M. (2018). Sales Forecasting: Machine Learning Solution to B2B Sales 
Opportunity Win-Propensity Computation. 
Lawrence, Rick, Perlich, Claudia, Rosset, Saharon, Khabibrakhmanov, Ildar,
"""
        ],

        # Retrieved context for Question 4
        [
            """Theoretical and experimental studies indicate that, besides an un -
equal class frequency distribution, the following factors influence the 
modeling of a capable classifier in identifying rare events (Sun et al., 
2009): 
• Sample size. When sample size is limited, discovering patterns cor -
responding to the minority class is unreliable. Experimental obser -
vations indicate that as the size of the training set increases, the error 
rate caused by the imbalanced class distribution decreases (Japko -
wicz & Stephen, 2002).  
• Class separability. Referring to the degree of discriminative patterns 
within classes. Research shows that the unequal frequency distri -
bution of instances among classes by itself is less worrisome, but 
combined with overlapping discriminative patterns between classes, 
it can significantly decrease the number of minority class instances 
correctly classified (Prati & Batista, 2004).

combined with overlapping discriminative patterns between classes, 
it can significantly decrease the number of minority class instances 
correctly classified (Prati & Batista, 2004).  
• Within-class imbalance. In many classification problems, a single class 
is composed of various subclasses. Within-class imbalance corre -
sponds to an imbalanced class distribution among subclasses and 
worsens the imbalance distribution problem in two ways: (1) 
increased learning complexity and (2) within-class subclasses are 
usually not apparent (Japkowicz, 2001). 
Fig. 1. Predicting B2B sales potential methodology (Bohanec et al., 2015b).  
D. Rohaan et al.

Expert
Systems With Applications 188 (2022) 115925
3
2.3. Automation using NLP 
Automation of B2B sales forecasting is vital for a number of reasons: 
achieving greater sales productivity and sales go-to-market alignment 
(Lawrence et al., 2010), improved planning (Lu & Kao, 2016), higher 
efficiency and sale prioritization (Duncan & Elkan, 2015), effective 
allocation of resources (D’Haen & Van den Poel, 2013) and under -
standing of the driving factors behind successful sale (Bohanec, 2017; 
Lambert, 2018). 
The solution to automated B2B sales forecasting lies in the field of 
Natural Language Processing, more specifically in its sub-field Infor -
mation Extraction. Information extraction concerns extraction of struc-
tured information from unstructured or semi-structured text in machine- 
readable documents (Jiang, 2012). Two fundamental tasks of informa -
tion extraction are named entity recognition (NER) and relation

can use to adjust their current mental decision models. 
Despite major progress within forecasting methods as a result of 
advancements in machine learning research, B2B sales forecasting 
methods experience little improvement (Bohanec et al., 2015b). Our 
research contributes to B2B sales forecasting in two ways. First, by 
giving step-by-step guidance on incorporating supervised machine 
learning in B2B sales forecasting and revealing potential pitfalls along 
the way, we seek to bridge the gap between theory and practice. Second, 
we give an indication of the performance improvement that can be ex -
pected when adopting supervised machine learning into B2B sales 
forecasting, showing its need for adoption if one is to stay ahead of 
competition. 
2.2. Imbalanced data classification 
Imbalanced data is characterized by an unequal frequency distribu-
tion of instances among classes and is present in our research since the

competition. 
2.2. Imbalanced data classification 
Imbalanced data is characterized by an unequal frequency distribu-
tion of instances among classes and is present in our research since the 
average ratio of an RFQ ever becoming a sale is only about 17%. Clas -
sification with imbalanced data has encountered a significant drawback 
of the performance achieved by most standard classifier algorithms 
which assume a relatively balanced class distribution and equal 
misclassification costs (Sun, Wong, & Kamel, 2009). Since minority class 
instances occur less frequent, classification rules predicting the minority 
class(es) tend to be rare, undiscovered or ignored and consequently, 
samples belonging to the minority class(es) are misclassified more often 
than those belonging to the majority class(es) (Sun et al., 2009). 
Theoretical and experimental studies indicate that, besides an un -
equal class frequency distribution, the following factors influence the

Theoretical and experimental studies indicate that, besides an un -
equal class frequency distribution, the following factors influence the 
modeling of a capable classifier in identifying rare events (Sun et al., 
2009): 
• Sample size. When sample size is limited, discovering patterns cor -
responding to the minority class is unreliable. Experimental obser -
vations indicate that as the size of the training set increases, the error 
rate caused by the imbalanced class distribution decreases (Japko -
wicz & Stephen, 2002).  
• Class separability. Referring to the degree of discriminative patterns 
within classes. Research shows that the unequal frequency distri -
bution of instances among classes by itself is less worrisome, but 
combined with overlapping discriminative patterns between classes, 
it can significantly decrease the number of minority class instances 
correctly classified (Prati & Batista, 2004).

can use to adjust their current mental decision models. 
Despite major progress within forecasting methods as a result of 
advancements in machine learning research, B2B sales forecasting 
methods experience little improvement (Bohanec et al., 2015b). Our 
research contributes to B2B sales forecasting in two ways. First, by 
giving step-by-step guidance on incorporating supervised machine 
learning in B2B sales forecasting and revealing potential pitfalls along 
the way, we seek to bridge the gap between theory and practice. Second, 
we give an indication of the performance improvement that can be ex -
pected when adopting supervised machine learning into B2B sales 
forecasting, showing its need for adoption if one is to stay ahead of 
competition. 
2.2. Imbalanced data classification 
Imbalanced data is characterized by an unequal frequency distribu-
tion of instances among classes and is present in our research since the

mining, inference, and prediction. Springer.  
Heinze, G., & Dunkler, D. (2016). Five myths about variable selection. Transplant 
International, 30(1), 1–6. 
Hirschberg, Julia, & Manning, Christopher D. (2015). Advances in natural language 
processing. Science, 349(6245), 261–266. 
Japkowicz, N. (2001). Concept-learning in the presence of between-class and within- 
class imbalances. Canadian Conference on AI. Springer. 
Japkowicz, N., & Stephen, S. (2002). The class imbalance problem: A systematic study. 
Intelligent Data Analysis, 31. 
Jiang, J. (2012). Information extraction from text. In C. C. Aggarwal, & C. (. Zhai, Mining 
text data. Springer. 
Kuhn, M., & Johnson, K. (2019). Feature Engineering and Selection: A Practical Approach for 
Predictive Models ((1 ed.).). CRC Press.  
Lambert, M. (2018). Sales Forecasting: Machine Learning Solution to B2B Sales 
Opportunity Win-Propensity Computation. 
Lawrence, Rick, Perlich, Claudia, Rosset, Saharon, Khabibrakhmanov, Ildar,

Expert
Systems With Applications 188 (2022) 115925
10
Investigation of flagged outlier samples in the ERP system of the service 
provider revealed that these were caused by human error, confirming 
the necessity. Under-sampling was carried out to address the class 
imbalance in the dataset (Sun et al., 2009) since the historical average 
ratio that an RFQ ever becomes a sale is only about 17%. In our dataset 
19% of quote lines converted into a sale. After under-sampling, the data 
used for training the model consists of 54,874 quote lines, equally 
distributed amongst classes ‘Sale’/’No Sale’. 
Next, we train the prediction model on the remaining 54,874 his -
torical quote lines. The prediction model, following from Section 3.5, is 
a 13-feature Random Forest model with hyper-parameter configuration 
as seen in Table 4 and a classification threshold of 0.62. Finally, we 
apply the trained prediction model to the testing data set and compare
"""
        ],

        # Retrieved context for Question 5
        [
            """
In our research, we use (random) stratified splitting, ensuring equal 
frequency distribution of classes in the data used for model training and 
-evaluation (Kuhn & Johnson, 2019). Moreover, we combine the strat -
ified splitting with k-fold cross validation to prevent overfitting. This 
means that data is split into k folds with equal class distribution, the 
model is trained on k-1 folds and evaluated on the remaining fold. This 
process is repeated for multiple splits, in which the training/testing fold 
allocation differ. The final performance estimation is the average per -
formance estimation of all k testing folds over all splits. 
3.4. Model training & building 
3.4.1. Outlier detection 
Outliers only exist in non-categorial features. To detect outliers, Z- 
scoring method is applied to the numerical features. This method as -

3.4. Model training & building 
3.4.1. Outlier detection 
Outliers only exist in non-categorial features. To detect outliers, Z- 
scoring method is applied to the numerical features. This method as -
sumes that values of the concerned features are normally distributed. In 
this method, any instance containing a numerical feature value more 
than 3 standard deviations away from the mean is discarded. 
3.4.2. Resampling 
Solutions to address class imbalance are adapting existing algo -
rithms, boosting, cost-sensitive learning and data resampling (Sun et al., 
2009). In our research we address class imbalance via boosting 
(considering the Gradient Boosting Classifier algorithm), cost-sensitive 
learning (indirectly via the F1-score performance measure) and data 
resampling. 
Regarding data resampling, for our case study, there are two ap -
proaches: under- and oversampling. This because we want both the 
majority- and minority class to have an equal recognition rate. Over -

proaches: under- and oversampling. This because we want both the 
majority- and minority class to have an equal recognition rate. Over -
sampling entails the replication/creation of new instances (usually mi -
nority class) from existing ones, and under-sampling the exclusion of 
instances (usually majority class). Between the two methods, over -
sampling yields the highest performance increase as it increases the 
minority class recognition rate without sacrificing the majority class 
recognition rate (Batuwita & Palade, 2010). Yet, there are also practical 
issues to take into consideration when choosing between both methods, 
such as model training time, especially when incorporating cross- 
validation. Therefore, we prefer under-sampling in our paper. 
3.5. Model improvement 
In our research, we focused our improvement efforts on metric F1- 
score, which represents the harmonic mean of precision, the percent -

No. estimators Min samples split Min samples leaf Max features Max depth Criterion Bootstrap 
400 2 1 13 None Gini False  
Fig. 9. Validation curves classification threshold optimization (k = 3).  
D. Rohaan et al.

Expert
Systems With Applications 188 (2022) 115925
10
Investigation of flagged outlier samples in the ERP system of the service 
provider revealed that these were caused by human error, confirming 
the necessity. Under-sampling was carried out to address the class 
imbalance in the dataset (Sun et al., 2009) since the historical average 
ratio that an RFQ ever becomes a sale is only about 17%. In our dataset 
19% of quote lines converted into a sale. After under-sampling, the data 
used for training the model consists of 54,874 quote lines, equally 
distributed amongst classes ‘Sale’/’No Sale’. 
Next, we train the prediction model on the remaining 54,874 his -
torical quote lines. The prediction model, following from Section 3.5, is 
a 13-feature Random Forest model with hyper-parameter configuration 
as seen in Table 4 and a classification threshold of 0.62. Finally, we 
apply the trained prediction model to the testing data set and compare

as seen in Table 4 and a classification threshold of 0.62. Finally, we 
apply the trained prediction model to the testing data set and compare 
the predicted class outcomes with the true known class outcomes. From 
this follows that the performance is 56.24% for the F1 score, 66.9% for 
the recall, 48.5% for the precision, 83% for the AUC and, following from 
the confusion matrix, ~80% of non-sales are correctly predicted 
(Fig. 10, Table 5). 
Thus, our model predicts 48.5% correctly to be a sale. Compared to 
19% quote-to-sales order conversion rate in our dataset, resulting from 
manual handling at the service provider, this represents a performance 
increase of + 155.3% (slightly more than a factor 2.5 times the perfor-
mance using manual handling). The need for the prediction model 
sprung, amongst other things, from the fact that the service provider 
receives more RFQs than it can process. Using the prediction model in

cess experts at the service provider, we differentiate between two ap -
proaches for handling MVs. First, instances with ≥ 1 MAR/MCAR 
feature values are discarded. We discard these instances because there is 
plenty of data and imputation, even a sophisticated method such as 
multiple imputation, would induce unnecessary uncertainty. Second, 
missing values of MNAR features are replaced by ‘unknown’. 
3.3. Data splitting 
Historically 17% of RFQs turn into a sale, meaning we are dealing 
with unequal distribution of instances amongst classes, i.e. class 
imbalance. When dealing with imbalanced data, it is best to take either a 
balanced- or stratified sample. Both types of samples force a certain 
composition to the to-be predicted target feature, where in a balanced 
sample this composition is predefined and in a stratified sample the 
composition follows the natural class distribution of the original data 
set. 
In our research, we use (random) stratified splitting, ensuring equal

mining, inference, and prediction. Springer.  
Heinze, G., & Dunkler, D. (2016). Five myths about variable selection. Transplant 
International, 30(1), 1–6. 
Hirschberg, Julia, & Manning, Christopher D. (2015). Advances in natural language 
processing. Science, 349(6245), 261–266. 
Japkowicz, N. (2001). Concept-learning in the presence of between-class and within- 
class imbalances. Canadian Conference on AI. Springer. 
Japkowicz, N., & Stephen, S. (2002). The class imbalance problem: A systematic study. 
Intelligent Data Analysis, 31. 
Jiang, J. (2012). Information extraction from text. In C. C. Aggarwal, & C. (. Zhai, Mining 
text data. Springer. 
Kuhn, M., & Johnson, K. (2019). Feature Engineering and Selection: A Practical Approach for 
Predictive Models ((1 ed.).). CRC Press.  
Lambert, M. (2018). Sales Forecasting: Machine Learning Solution to B2B Sales 
Opportunity Win-Propensity Computation. 
Lawrence, Rick, Perlich, Claudia, Rosset, Saharon, Khabibrakhmanov, Ildar,

Mortensen, S., Christison, M., Li, B., Zhu, A., & Venkatesan, R. (2019). Predicting and 
Defining B2B Sales Success with Machine Learning. Charlottesville: IEEE.  
Orange Data Mining. (2020). Retrieved from https://orangedatamining.com. 
Pinçe, Ç., Turrini, L., & Meissner, J. (2021). Intermittent Demand Forecasting for Spare 
Parts: A Critical Review. Omega, 1–30. 
Prati, R. C., & Batista, G. E. (2004). Class imbalances versus class overlapping: an 
analysis of a learning system behavior. Mexican International Conference on 
Artificial Intelligence. Springer. 
Sun, Y., Wong, A. K., & Kamel, M. S. (2009). Classification of imbalanced data: A review. 
International Journal of Pattern Recognition and Artificial Intelligence, 3. 
Syntetos, Aris A., & Boylan, John E. (2005). The accuracy of intermittent demand 
estimates. International Journal of forecasting, 21(2), 303–314. 
Tan, P.-N., Steinbach, M., & Kumar, V. (2006). Classification: Basic Concepts, Decision
"""
        ],

        # Retrieved context for Question 6
        [
            """formance by + 155.3% over prior manual handling and demonstrated 
the feasibility of automation through a proof of concept. The need for 
the sales forecasting model sprung, amongst other things, from the fact 
that the service provider receives more RFQs than it is able to process. 
Using the model in the future will allow the service provider to generate 
more sales, assuming the same distribution of ‘Sale’/’No sale’ amongst 
the RFQs that previously could not be processed, by responding to RFQs 
in order of descending predicted probability of sale. 
Regarding the limitations and future research direction, first, our 
prediction model predicts solely sales potential but neglects the value of 
the corresponding RFQ. This means that a low-cost part (e.g. bolt) could 
be prioritized over a high value part (e.g. engine) with neglectable dif-
ference in sales potential. Second, in our research we focused on opti -
mizing the F1 score, the harmonic mean between precision and recall.

ference in sales potential. Second, in our research we focused on opti -
mizing the F1 score, the harmonic mean between precision and recall. 
Table 6 
Features used by the 13-feature Random Forest model to predict sales potential.  
Rank Feature name Feature description 
1 Hit rate account The percentage of the total quoted value for an 
account that converted to a sale at the time that the 
quote was issued. 
2 Frequency account Frequency count of an account number, and thus 
customer, in the sales order data at the time that the 
quote was issued. 
3 Main Supplier List 
Price 
Unnegotiated supplier part price which is visible to 
the entire world. This feature is an indication of the 
purchase/sales price as the service provider can 
sometimes negotiate a better price. 
4 Frequency part Frequency count of a part number, and thus part, in 
the sales order data at the time that the quote was 
issued. 
5 Account (ABC) This feature represents ABC classified feature 
account number.

the sales order data at the time that the quote was 
issued. 
5 Account (ABC) This feature represents ABC classified feature 
account number. 
6 Customer type The customer company type. Note: different sales 
managers might categorize customers differently as 
there are no categorization rules. 
7 Stock Whether the requested part was on stock. This 
feature was approximated in consultation with 
process experts at the service provider. Here, we 
assume that if a part was on stock, its delivery 
window would be ≤ 7 days. 
8 Part (ABC) This feature represents ABC classified feature part 
number. 
9 Supplier (ABC) This feature represents ABC classified feature 
(default) supplier ID. 
10 Part manufacturer 
(ABC) 
This feature represents ABC classified feature part 
manufacturer ID. 
11 Rotable part Binary feature representing whether the part is a 
rotable part or not. Rotable parts are parts that can 
be repaired (second-hand). 
12 Sales manager The regional sales manager.

Sale=1.0 
19 0.111 0.619 1.237 Account (ABC)=(80, 100], 
Supplier type=Unknown, 
Rotable part=0 
Sale=0.0 
20 0.108 0.848 1.696 Account (ABC)=(80,100], 
Customer type=BROKER 
Sale=0.0 
21 0.108 0.679 2.729 Customer type=BROKER Account 
(ABC)=(80, 
100], Sale=0.0 
22 0.106 0.653 1.307 Customer type=DEFENSE, 
Part (ABC)=(80,100] 
Sale=1.0 
23 0.103 0.663 1.326 Customer type=AIRLINE, 
Rotable part=0, Stock=1.0 
Sale=1.0 
24 0.103 0.664 1.328 Customer type=DEFENSE, 
Part (ABC)=(80,100], 
Rotable part=0 
Sale=1.0 
25 0.103 0.64 1.561 Customer type=DEFENSE 
Part (ABC)=(80, 100] 
Rotable part=0, 
Sale=1.0  
D. Rohaan et al.

Expert
Systems With Applications 188 (2022) 115925
12
Yet, in practice it is unlikely that a missed sale and wasted employee 
time are equally valuable, and therefore investigation into an optimal 
(subjective) precision-recall trade-off is recommended. Third, we 
recommend retraining the model regularly and monitoring performance 
over time. Feature selection, hyper-parameter tuning and classification 
threshold optimization were statically executed. If performance drops 
over time, e.g., potentially due to change in importance of features, our 
research should be repeated. Fourth, our model can only be applied to 
RFQs from existing customers, concerning parts in the product range of 
the service provider. We elaborate further on this limitation in Section 6. 
Finally, the NLP POC falsely recognizes certain numbers occurring in the 
RFQs as part numbers. These numbers were actual part numbers but not 
in the context of the RFQ (e.g. street number of the service provider). To

RFQs as part numbers. These numbers were actual part numbers but not 
in the context of the RFQ (e.g. street number of the service provider). To 
diminish the extend of this issue we created an array in which frequently 
falsely identified (part) numbers can be specified, which will then not be 
fed to the Entity Ruler and therefore no longer (falsely) recognized. 
Here, we recommend the service provider to investigate the revenue 
corresponding to such frequently falsely identified parts and decide 
whether to exclude them. 
6. Ethical considerations 
AI presents three major areas of ethical concern for society: privacy 
and surveillance, bias and discrimination, and perhaps the deepest, most 
difficult philosophical question of the era, the role of human judgment 
(xThe Harvard Gazette, 2020). In our research we encounter ethical 
considerations in the latter two areas, which we discuss below. 
First, the prediction model prioritizes RFQs based on estimated sales

senting the requested quantity for a specific part by the corresponding 
customer. Consequently, each quote line may or may not convert into a 
sales order. 
RFQs that were not pursued by the service provider cannot be 
considered in our research since these lack a class label (Section 3). 
Therefore, we work with quote line data in our research. The inevitable 
consequence of this is that the input data, and therefore the model and 
predictions will initially contain a bias induced through earlier RFQ 
assessments of the service provider. 
3.1.2. Data set 
The data set used for training- and evaluating our model covers 
features from multiple data sources, describing the customer, the 
requested part, the (requested) part vendor, the corresponding (or non- 
existing) sales order and concerned quote, linked together on primary 
keys. Note, primary key features are unique identifiers and therefore 
solely used to link data sources rather than for model learning.

representation based on historical sales data. This step requires identi -
fying feature data describing a sales opportunity and deriving additional 
custom features, so called ‘meta variables’ (Mortensen, Christison, Li, 
Zhu, & Venkatesan, 2019). The second step is the data preparation, 
entailing data cleaning, -transformation and -splitting to prevent 
‘Garbage in, Garbage out’. The objective of the third step is to identify 
the model that is best in predicting sales potential. Here, we extended 
the existing methodology by incorporating a feature selection method -
ology (Bohanec et al., 2015a), hyperparameter tuning and classification 
threshold optimization. The final step uses ML techniques and 
visualizations to gain/emphasize insights which the sales department 
can use to adjust their current mental decision models. 
Despite major progress within forecasting methods as a result of 
advancements in machine learning research, B2B sales forecasting

of the search space in an exponential manner. Feature selection is used 
to address the curse of dimensionality. This speeds up computation time, 
improves input data quality, and potentially increases model perfor -
mance while simultaneously decreasing model complexity. We use a 
feature selection methodology by Bohanec et al. (2015a). This method 
consists of the following steps:  
1) Ranking features according to importance.  
2) Adding the features with the highest importance and monitoring 
model performance each time until the optimal cut-off point is 
obtained.  
3) Eliminating noise/redundancy using a wrapper method. 
It should be noted that literature recommends feature selection to be 
carried out only if the number of events per variable (EPV), which is the 
smallest of the number of positive/negative cases (sales/non-sales), 
divided by the number of independent features, is at least 50 (Heinze & 
Dunkler, 2016). In the training data, used below for feature selection,
"""
        ],

        # Retrieved context for Question 7
        [
            """senting the requested quantity for a specific part by the corresponding 
customer. Consequently, each quote line may or may not convert into a 
sales order. 
RFQs that were not pursued by the service provider cannot be 
considered in our research since these lack a class label (Section 3). 
Therefore, we work with quote line data in our research. The inevitable 
consequence of this is that the input data, and therefore the model and 
predictions will initially contain a bias induced through earlier RFQ 
assessments of the service provider. 
3.1.2. Data set 
The data set used for training- and evaluating our model covers 
features from multiple data sources, describing the customer, the 
requested part, the (requested) part vendor, the corresponding (or non- 
existing) sales order and concerned quote, linked together on primary 
keys. Note, primary key features are unique identifiers and therefore 
solely used to link data sources rather than for model learning.

keys. Note, primary key features are unique identifiers and therefore 
solely used to link data sources rather than for model learning. 
Furthermore, note that quote features cannot be used for learning since 
these are not available at the point in time an RFQ arrives. 
Fig. 2. Supervised Machine Learning framework.  
D. Rohaan et al.

Expert
Systems With Applications 188 (2022) 115925
4
We created 5 additional custom features (meta-features), as example 
cases of B2B sales prediction with ML have shown that these can 
potentially capture influence which the default features do not, and can 
be significant predictors (Mortensen et al., 2019). These meta features 
are the following: 
Frequency part, frequency count of a part number, and thus part, in 
the sales order data at the time that the quote was issued. 
Frequency customer, frequency count of an account number, and thus 
customer, in the sales order data at the time that the quote was issued. 
Hit rate account, the percentage of the total quoted value for a 
customer account that converted to a sale at the time that the quote was 
issued. 
Hit rate part, the percentage of the total quoted value for a part that 
converted to a sale at the time that the quote was issued. 
Stock, whether the requested part is on stock. This meta-feature was

different classifiers are trained on different feature subsets of the 
training data and afterwards applied to the testing data. Accordingly, a 
performance estimation can be derived by comparing the predicted 
outcomes of the testing data with the true known outcomes of the testing 
data. Finally, the combination of feature subset and classifier that yields 
the best prediction performance is adopted and applied to future/unseen 
cases. 
3.1. Data acquisition 
3.1.1. From RFQ to sales order 
The process from RFQ to sales order at the service provider is shown 
in Fig. 3. If an RFQ is deemed to have the potential to result in a sale, a 
quote is created containing the offer (price, lead- time, etc.) in response 
to the RFQ. A quote may contain multiple (quote) lines, each repre -
senting the requested quantity for a specific part by the corresponding 
customer. Consequently, each quote line may or may not convert into a 
sales order.

senting the requested quantity for a specific part by the corresponding 
customer. Consequently, each quote line may or may not convert into a 
sales order. 
RFQs that were not pursued by the service provider cannot be 
considered in our research since these lack a class label (Section 3). 
Therefore, we work with quote line data in our research. The inevitable 
consequence of this is that the input data, and therefore the model and 
predictions will initially contain a bias induced through earlier RFQ 
assessments of the service provider. 
3.1.2. Data set 
The data set used for training- and evaluating our model covers 
features from multiple data sources, describing the customer, the 
requested part, the (requested) part vendor, the corresponding (or non- 
existing) sales order and concerned quote, linked together on primary 
keys. Note, primary key features are unique identifiers and therefore 
solely used to link data sources rather than for model learning.

keys. Note, primary key features are unique identifiers and therefore 
solely used to link data sources rather than for model learning. 
Furthermore, note that quote features cannot be used for learning since 
these are not available at the point in time an RFQ arrives. 
Fig. 2. Supervised Machine Learning framework.  
D. Rohaan et al.

3.7.1. Discrepancy RFQ data and customer data source 
The prediction model takes input features which can all be acquired 
if the account number (customer) and (requested) part number are 
known. Each RFQ contains, amongst other things, the requested part 
number(s), the name of the requesting company and the customer email 
address, where the latter two can theoretically both be linked to the 
feature account number. However, in practice there appears to be a 
significant discrepancy between customer company names as they occur 
in the customer data source and as they occur in the RFQs consisting of 
different wording rather than punctuation, capital letters, etc. Further -
more, the service provider did not collect/store customer email address 
data, corresponding to the RFQs received, due to which this information 
cannot be used for linking. Consequently, until either one of these issues 
is resolved, prioritization cannot happen automatically nor 
autonomously.

customer email address is extracted from the sender email address and 
for the indirect RFQs the partial customer email address is extracted 
from the RFQ text body. 
4. Results 
In this section we evaluate the performance of both the prediction 
model and NLP POC and provide insights into the prediction model. 
4.1. Prediction model 
The creation of the prediction model started with the collection of a 
dataset consisting of ~ 180,000 historical quote lines from 2012 to 
2019, describing the features shown in Fig. 4. Next, data cleaning took 
place, comprising of feature encoding and handling missing values 
(Section 3.2). Features were encoded using either label encoding, OHE 
or our custom ABC encoding, depending on their number of categorical 
values. Quote lines with ≥ 1 MAR/MCAR feature values were discarded, 
whereas MNAR feature values were replaced by ‘unknown’. The 
maximum percentage of missing values amongst MAR/MCAR features is

Expert
Systems With Applications 188 (2022) 115925
10
Investigation of flagged outlier samples in the ERP system of the service 
provider revealed that these were caused by human error, confirming 
the necessity. Under-sampling was carried out to address the class 
imbalance in the dataset (Sun et al., 2009) since the historical average 
ratio that an RFQ ever becomes a sale is only about 17%. In our dataset 
19% of quote lines converted into a sale. After under-sampling, the data 
used for training the model consists of 54,874 quote lines, equally 
distributed amongst classes ‘Sale’/’No Sale’. 
Next, we train the prediction model on the remaining 54,874 his -
torical quote lines. The prediction model, following from Section 3.5, is 
a 13-feature Random Forest model with hyper-parameter configuration 
as seen in Table 4 and a classification threshold of 0.62. Finally, we 
apply the trained prediction model to the testing data set and compare
"""
        ],

        # Retrieved context for Question 8
        [
            """There exist cases in which a sales order is created without a quote line. 
However, these are rare. Yet, there exists no unique identifier which a 
quote line and sales order have in common. Therefore, quote-sales order 
linking is approximated by requiring an account number-, part number- 
and sales type feature value combination match as well as a valid 
timeline in terms of both chronological order and the number of days 
between issuance of quote and sales order. Regarding the latter, we 
make a distinction between regular customers and governments, where 
the subsequent is allowed more time due to bureaucratic approvals. 
3.1.3.2. Linking customer- and part number data to quotes. Customer/ 
part- and quote data sources are linked on features account number/part 
number respectively, representing unique customers/parts, with a 
relation one to one-or-many. Yet, this link is valid only when the 
customer- and part in the quote occur in the customer- and part data

relation one to one-or-many. Yet, this link is valid only when the 
customer- and part in the quote occur in the customer- and part data 
source, which by their design only contain customers/parts that have 
been sold (to) at least once in the past. Hence, when a quote contains a 
customer/part that has not been sold in the past, all feature values will 
be missing, and by the design of the data sources, this quote will not 
have converted into a sale and this poses a data leak. We eliminate this 
leak by removing all rows for which all feature values of the customer- 
and/or part data source is missing. 
3.2. Data cleaning 
After establishing the ML data set, data cleaning took place 
comprising encoding of categorical features and handling missing 
values. 
3.2.1. Encoding categorical features 
Categorical features were encoded using either label encoding, one 
hot encoding (OHE) or our own developed ABC encoding, depending on

values. 
3.2.1. Encoding categorical features 
Categorical features were encoded using either label encoding, one 
hot encoding (OHE) or our own developed ABC encoding, depending on 
their number of categorical values (Zheng & Casari, 2018). 
Categorical features with two categorical values were label encoded, 
where one categorical value gets replaced with ‘0′ and the other with ‘1′. 
Be aware that, when applied to features with more than two categorical 
values, label coding misunderstands data to be in some kind of order. 
Instead, categorical features with more than two categorical values were 
one hot encoded, where the original categorical feature is transformed 
into a number of dummy features equal to the number of categorical 
values within the original categorical feature. Here, each categorical 
value is represented by a dummy feature containing ‘0′ and ’1′ entries 
depending on the presence of the categorical value. Note that OHE

dimensional matrix, where n is equal to the number of classes, indexed 
in one axis by the true class of an observation and in the other axis by the 
class that the classifier assigns. F1-score is the harmonic mean of the 
precision and the recall. Finally, the AUC, which stands for Area Under 
the ROC (Receiver Operating Characteristics) Curve, indicates the extent 
to which the prediction model can distinguish between classes. The AUC 
metric takes values on the interval [0.5, 1], where a score of 1 means the 
prediction model can perfectly distinguish between classes and a score 
of 0.5 means the prediction model cannot differentiate between classes. 
3.7. Model automation 
In this section we investigate automation of our prediction model 
and corresponding sales forecasts. 
3.7.1. Discrepancy RFQ data and customer data source 
The prediction model takes input features which can all be acquired 
if the account number (customer) and (requested) part number are

3.7.1. Discrepancy RFQ data and customer data source 
The prediction model takes input features which can all be acquired 
if the account number (customer) and (requested) part number are 
known. Each RFQ contains, amongst other things, the requested part 
number(s), the name of the requesting company and the customer email 
address, where the latter two can theoretically both be linked to the 
feature account number. However, in practice there appears to be a 
significant discrepancy between customer company names as they occur 
in the customer data source and as they occur in the RFQs consisting of 
different wording rather than punctuation, capital letters, etc. Further -
more, the service provider did not collect/store customer email address 
data, corresponding to the RFQs received, due to which this information 
cannot be used for linking. Consequently, until either one of these issues 
is resolved, prioritization cannot happen automatically nor 
autonomously.

cannot be used for linking. Consequently, until either one of these issues 
is resolved, prioritization cannot happen automatically nor 
autonomously. 
However, to demonstrate the feasibility/opportunity we decided to 
create a proof of concept (POC) for the automation part instead. Within 
this POC we argue for the use of partial customer email address data, 
from ‘@’ until the end of the email address, which primarily contains 
the customer company name. The reason for this being that a partial 
customer email addresses, representing the company email address 
domain, will rarely change over time, whereas customer company name 
can differ (or lack) in each RFQ and is prone to human error. 
3.7.2. NLP proof of concept 
The service provider RFQ email inbox is accessed after which the 
most recent emails are retrieved and put in a data frame consisting of 
columns containing their corresponding conversation ids, sender email 
addresses and text bodies.

senting the requested quantity for a specific part by the corresponding 
customer. Consequently, each quote line may or may not convert into a 
sales order. 
RFQs that were not pursued by the service provider cannot be 
considered in our research since these lack a class label (Section 3). 
Therefore, we work with quote line data in our research. The inevitable 
consequence of this is that the input data, and therefore the model and 
predictions will initially contain a bias induced through earlier RFQ 
assessments of the service provider. 
3.1.2. Data set 
The data set used for training- and evaluating our model covers 
features from multiple data sources, describing the customer, the 
requested part, the (requested) part vendor, the corresponding (or non- 
existing) sales order and concerned quote, linked together on primary 
keys. Note, primary key features are unique identifiers and therefore 
solely used to link data sources rather than for model learning.

customer email address is extracted from the sender email address and 
for the indirect RFQs the partial customer email address is extracted 
from the RFQ text body. 
4. Results 
In this section we evaluate the performance of both the prediction 
model and NLP POC and provide insights into the prediction model. 
4.1. Prediction model 
The creation of the prediction model started with the collection of a 
dataset consisting of ~ 180,000 historical quote lines from 2012 to 
2019, describing the features shown in Fig. 4. Next, data cleaning took 
place, comprising of feature encoding and handling missing values 
(Section 3.2). Features were encoded using either label encoding, OHE 
or our custom ABC encoding, depending on their number of categorical 
values. Quote lines with ≥ 1 MAR/MCAR feature values were discarded, 
whereas MNAR feature values were replaced by ‘unknown’. The 
maximum percentage of missing values amongst MAR/MCAR features is

cess experts at the service provider, we differentiate between two ap -
proaches for handling MVs. First, instances with ≥ 1 MAR/MCAR 
feature values are discarded. We discard these instances because there is 
plenty of data and imputation, even a sophisticated method such as 
multiple imputation, would induce unnecessary uncertainty. Second, 
missing values of MNAR features are replaced by ‘unknown’. 
3.3. Data splitting 
Historically 17% of RFQs turn into a sale, meaning we are dealing 
with unequal distribution of instances amongst classes, i.e. class 
imbalance. When dealing with imbalanced data, it is best to take either a 
balanced- or stratified sample. Both types of samples force a certain 
composition to the to-be predicted target feature, where in a balanced 
sample this composition is predefined and in a stratified sample the 
composition follows the natural class distribution of the original data 
set. 
In our research, we use (random) stratified splitting, ensuring equal
"""
        ],

        # Retrieved context for Question 9
        [
            """parameter values from the grid of the exploratory search are excluded 
in the targeted search if their frequency of occurrence in the top 20 
configurations (Table 13) is below a certain relative threshold. The grid 
values of the exploratory- and targeted random grid search are shown in 
Table 3. Fig. 8 shows the average F1 score (3-fold CV) ordered against 
the number of iterations of the exploratory- and targeted random search. 
Here, we see a steep- and flat improvement for the exploratory- and 
targeted random search respectively, confirming the need of an initial 
exploratory random search and suggesting that increasing the number of 
iterations in the targeted random search is not likely to result in a better 
configuration. 
From our hyper-parameter tuning methodology followed that the 
hyper-parameter configuration shown in Table 4 results in the highest 
performance. 
3.5.3. Classification threshold optimization

From our hyper-parameter tuning methodology followed that the 
hyper-parameter configuration shown in Table 4 results in the highest 
performance. 
3.5.3. Classification threshold optimization 
Classification is based on a probability that the classifier assigns to an 
observation using the inferences learned from the training data set. By 
default, the classification threshold is set to 0.5 meaning in this case, that 
an observation with a predicted probability greater than 0.5 is classified 
as ‘Sale’ and ≤ 0.5 as ‘No sale’. Yet, the classification threshold can be 
altered and therefore optimized. The classification threshold can be 
perceived as a hyper-parameter and should be treated accordingly. From 
the validation curves plotted in Fig. 9, we find that the optimal threshold 
lies within [0.55, 0.65]. After zooming-in on this area, we determine 
that the optimal classification threshold, given the optimal feature 
subset and -hyper parameter configuration, is 0.62.

lies within [0.55, 0.65]. After zooming-in on this area, we determine 
that the optimal classification threshold, given the optimal feature 
subset and -hyper parameter configuration, is 0.62. 
3.6. Model testing 
We selected five performance measures to evaluate the performance 
of our prediction model. These are recall, precision, confusion matrix, 
F1-score, and AUC, and are explained in the remainder of this sub- 
section (Tharwat, 2020). Precision is the percentage of correctly pre -
dicted sales over the total number of (correctly or falsely) predicted 
sales. Recall is the percentage of correctly predicted sales over the total 
number of actual sales. A confusion matrix summarizes the classification 
performance of a classifier with respect to some test data. It is a n- 
dimensional matrix, where n is equal to the number of classes, indexed 
in one axis by the true class of an observation and in the other axis by the

No. estimators Min samples split Min samples leaf Max features Max depth Criterion Bootstrap 
400 2 1 13 None Gini False  
Fig. 9. Validation curves classification threshold optimization (k = 3).  
D. Rohaan et al.

Expert
Systems With Applications 188 (2022) 115925
10
Investigation of flagged outlier samples in the ERP system of the service 
provider revealed that these were caused by human error, confirming 
the necessity. Under-sampling was carried out to address the class 
imbalance in the dataset (Sun et al., 2009) since the historical average 
ratio that an RFQ ever becomes a sale is only about 17%. In our dataset 
19% of quote lines converted into a sale. After under-sampling, the data 
used for training the model consists of 54,874 quote lines, equally 
distributed amongst classes ‘Sale’/’No Sale’. 
Next, we train the prediction model on the remaining 54,874 his -
torical quote lines. The prediction model, following from Section 3.5, is 
a 13-feature Random Forest model with hyper-parameter configuration 
as seen in Table 4 and a classification threshold of 0.62. Finally, we 
apply the trained prediction model to the testing data set and compare

as seen in Table 4 and a classification threshold of 0.62. Finally, we 
apply the trained prediction model to the testing data set and compare 
the predicted class outcomes with the true known class outcomes. From 
this follows that the performance is 56.24% for the F1 score, 66.9% for 
the recall, 48.5% for the precision, 83% for the AUC and, following from 
the confusion matrix, ~80% of non-sales are correctly predicted 
(Fig. 10, Table 5). 
Thus, our model predicts 48.5% correctly to be a sale. Compared to 
19% quote-to-sales order conversion rate in our dataset, resulting from 
manual handling at the service provider, this represents a performance 
increase of + 155.3% (slightly more than a factor 2.5 times the perfor-
mance using manual handling). The need for the prediction model 
sprung, amongst other things, from the fact that the service provider 
receives more RFQs than it can process. Using the prediction model in

as seen in Table 4 and a classification threshold of 0.62. Finally, we 
apply the trained prediction model to the testing data set and compare 
the predicted class outcomes with the true known class outcomes. From 
this follows that the performance is 56.24% for the F1 score, 66.9% for 
the recall, 48.5% for the precision, 83% for the AUC and, following from 
the confusion matrix, ~80% of non-sales are correctly predicted 
(Fig. 10, Table 5). 
Thus, our model predicts 48.5% correctly to be a sale. Compared to 
19% quote-to-sales order conversion rate in our dataset, resulting from 
manual handling at the service provider, this represents a performance 
increase of + 155.3% (slightly more than a factor 2.5 times the perfor-
mance using manual handling). The need for the prediction model 
sprung, amongst other things, from the fact that the service provider 
receives more RFQs than it can process. Using the prediction model in

lies within [0.55, 0.65]. After zooming-in on this area, we determine 
that the optimal classification threshold, given the optimal feature 
subset and -hyper parameter configuration, is 0.62. 
3.6. Model testing 
We selected five performance measures to evaluate the performance 
of our prediction model. These are recall, precision, confusion matrix, 
F1-score, and AUC, and are explained in the remainder of this sub- 
section (Tharwat, 2020). Precision is the percentage of correctly pre -
dicted sales over the total number of (correctly or falsely) predicted 
sales. Recall is the percentage of correctly predicted sales over the total 
number of actual sales. A confusion matrix summarizes the classification 
performance of a classifier with respect to some test data. It is a n- 
dimensional matrix, where n is equal to the number of classes, indexed 
in one axis by the true class of an observation and in the other axis by the

values. Quote lines with ≥ 1 MAR/MCAR feature values were discarded, 
whereas MNAR feature values were replaced by ‘unknown’. The 
maximum percentage of missing values amongst MAR/MCAR features is 
3.5% and there are two MNAR features with ~ 70% and ~ 35% MVs. 
Then, data was split using stratified k-fold cross validation (k = 5) in a 
training- and testing data set. We have used splitting ratio 80/20 for 
training- and testing data respectively. Afterwards, outlier detection and 
under-sampling were carried out on the training data (Section 3.4). 
Fig. 8. Results exploratory- (left) and targeted (right) random grid search. CV stands for cross-validation.  
Table 4 
Best performing hyper-parameter configuration.  
Hyper-parameter configuration 13-feature Random Forest model 
No. estimators Min samples split Min samples leaf Max features Max depth Criterion Bootstrap 
400 2 1 13 None Gini False  
Fig. 9. Validation curves classification threshold optimization (k = 3).

"""
        ],

        # Retrieved context for Question 10
        [
            """readable documents (Jiang, 2012). Two fundamental tasks of informa -
tion extraction are named entity recognition (NER) and relation 
extraction. First, a named entity is a sequence of words that refers to a 
real-world entity. NER identifies named entities from unstructured text 
and classifies them into a set of predefined types. Usually, NER cannot be 
simply accomplished by string/pattern matching against pre-compiled 
dictionaries, so-called entity ruler, because (1) named entities usually 
do not form a closed set and (2) named entities can be context depen -
dent. In such cases a NER model could provide a solution, which iden -
tifies and categorizes entities in unstructured data, on the basis of 
unstructured labelled training data. Second, relation extraction is the 
task of detecting and characterizing the semantic relations between 
entities in text, which is less relevant for our research objective. 
2.4. Spare parts demand forecasting

task of detecting and characterizing the semantic relations between 
entities in text, which is less relevant for our research objective. 
2.4. Spare parts demand forecasting 
There are several papers on forecasting spare parts demand. The 
main focus of this stream of research is to forecast slow moving erratic, 
lumpy, or intermittent demand patterns, which are typical characteris -
tics of spare parts demand. One of the seminal works is Croston (1972). 
Several other papers propose approaches to extend this work e.g., Syn-
tetos and Boylan (2005), Teunter, Syntetos, and Babai (2011) and Babai, 
Dallery, Boubaker, and Kalai (2019). We refer to Boylan and Syntetos 
(2010) and Pinçe, Turrini, and Meissner (2021) for reviews. Although 
our aim is to forecast spare parts demand also, we differ from this 
literature in three ways:  
1. Unlike these papers, who focus on point forecast of demand using 
demand historical sales data, we estimate the likelihood of an RFQ to 
be successful.

literature in three ways:  
1. Unlike these papers, who focus on point forecast of demand using 
demand historical sales data, we estimate the likelihood of an RFQ to 
be successful.  
2. In this regard, we focus on the likelihood of each individual RFQ, and 
therefore, our predictions are coupled with each individual piece of 
information (RFQ).  
3. Therefore, our methodology is also different. Unlike those paper on 
forecasting spare parts, which predominantly use time series models, 
we use machine learning and natural language processing to forecast 
spare part demand based on unstructured data retrieved from RFQs. 
Furthermore, particularly because of (1) and (2), our paper is much 
closer to advance demand information (ADI) since RFQs can be 
considered ADI (Topan, Tan, van Houtum, & Dekker, 2018). Yet, 
different from those papers on ADI, we are presenting a case study to 
show how RFQs can be used as an ADI and how this information is used

task of detecting and characterizing the semantic relations between 
entities in text, which is less relevant for our research objective. 
2.4. Spare parts demand forecasting 
There are several papers on forecasting spare parts demand. The 
main focus of this stream of research is to forecast slow moving erratic, 
lumpy, or intermittent demand patterns, which are typical characteris -
tics of spare parts demand. One of the seminal works is Croston (1972). 
Several other papers propose approaches to extend this work e.g., Syn-
tetos and Boylan (2005), Teunter, Syntetos, and Babai (2011) and Babai, 
Dallery, Boubaker, and Kalai (2019). We refer to Boylan and Syntetos 
(2010) and Pinçe, Turrini, and Meissner (2021) for reviews. Although 
our aim is to forecast spare parts demand also, we differ from this 
literature in three ways:  
1. Unlike these papers, who focus on point forecast of demand using 
demand historical sales data, we estimate the likelihood of an RFQ to 
be successful.

literature in three ways:  
1. Unlike these papers, who focus on point forecast of demand using 
demand historical sales data, we estimate the likelihood of an RFQ to 
be successful.  
2. In this regard, we focus on the likelihood of each individual RFQ, and 
therefore, our predictions are coupled with each individual piece of 
information (RFQ).  
3. Therefore, our methodology is also different. Unlike those paper on 
forecasting spare parts, which predominantly use time series models, 
we use machine learning and natural language processing to forecast 
spare part demand based on unstructured data retrieved from RFQs. 
Furthermore, particularly because of (1) and (2), our paper is much 
closer to advance demand information (ADI) since RFQs can be 
considered ADI (Topan, Tan, van Houtum, & Dekker, 2018). Yet, 
different from those papers on ADI, we are presenting a case study to 
show how RFQs can be used as an ADI and how this information is used

different from those papers on ADI, we are presenting a case study to 
show how RFQs can be used as an ADI and how this information is used 
to estimate the success probability that an RFQ will become a sale. 
3. B2B sales potential forecasting model 
In this section we explain the methodology used to create the ma -
chine learning model step-by-step, structured according to the frame -
work of supervised machine learning (Fig. 2). 
Supervised machine learning (Fig. 2) starts by collecting historical 
labelled data (data whose class is known) of instances, reflecting fea -
tures (that could be) of importance in predicting class label. Then, data 
cleaning takes place after which the cleaned data is split into a training- 
and testing data set (usually ratio 70/30 or 80/20 respectively). Next, 
different classifiers are trained on different feature subsets of the 
training data and afterwards applied to the testing data. Accordingly, a

reasons, consequently making performance seem better than it actually 
is and complete loss of information due to the wrong choice of encoding. 
The remainder of the paper is structured as follows: Section 2 pre-
sents the theoretical background and discusses our contribution to 
literature. Section 3 discusses our method. The results are presented in 
Section 4. In Section 5 we conclude our research and discuss the main 
findings of our work, indicate the limitations, and lay the foundation for 
further research. Finally, we discuss the ethical considerations of our 
research in Section 6. 
2. Literature review 
This section provides an overview of prior work relevant to achieving 
our research objective; B2B sales forecasting, imbalanced data classifi -
cation, automation using NLP and spare part demand forecasting, 
including studies on advance demand information (ADI). 
2.1. Business-to-business sales potential forecasting

cation, automation using NLP and spare part demand forecasting, 
including studies on advance demand information (ADI). 
2.1. Business-to-business sales potential forecasting 
Bohanec et al. (2015b) proposes a methodology for incorporating 
supervised machine learning in B2B sales forecasting (Fig. 1). Super -
vised Machine Learning is a variation of the machine learning paradigm 
where a classifier maps feature data, describing measurable properties 
or characteristics of a phenomenon being observed/analyzed, onto a 
class label. Here, a classifier is defined as an algorithm that identifies to 
which class an observation belongs, on the basis of a training data set 
containing (historical) observations whose class labels are known 
(Aggarwal, 2014). 
The first step in this methodology is to create a sales opportunity 
representation based on historical sales data. This step requires identi -
fying feature data describing a sales opportunity and deriving additional

future demand information coming from customers, taking the form of 
requests for quotation (RFQs). RFQs are uncommitted requests for a quote 
of spare parts and/or exchange of parts, by means of email containing 
unstructured text, that do not necessarily result in a sale. Despite the fact 
that they are uncommitted, RFQs can be used to predict future demand 
using artificial intelligence techniques, e.g. supervised machine learning 
and natural language processing. 
The research in this paper is a case study carried out at a large after- 
sales service and maintenance provider, to which we will refer as the 
service provider. The service provider receives a large number of RFQs. 
Yet, the average ratio that an RFQ ever becomes a sale is only about 
17%. Furthermore, these large number of RFQs exceed capacity of em-
ployees responsible for responding to the RFQs. This increases the 
respond time of the service provider to an RFQ, which is an important
"""
        ]
    ],

    "ground_truth": [
        # Reference answer for Question 1
        "The main objective is to use supervised machine learning and natural language processing to predict whether an RFQ will result in a sale and to prioritize RFQs according to their sales potential.",

        # Reference answer for Question 2
        "RFQ stands for Request for Quotation. Historically, approximately 17% of RFQs resulted in a sale.",

        # Reference answer for Question 3
        "A Random Forest classification model with 13 features, 400 trees, and a classification threshold of 0.62 was selected as the final model.",

        # Reference answer for Question 4
        "It is imbalanced because only about 17% of RFQs resulted in sales, while the majority did not result in sales.",

        # Reference answer for Question 5
        "The study investigated boosting, cost-sensitive learning, and resampling techniques. The final approach used under-sampling to balance the classes.",

        # Reference answer for Question 6
        "Three important features were the account hit rate, account frequency, and the main supplier list price.",

        # Reference answer for Question 7
        "Quote features were not available at the moment an RFQ was received. Using them would therefore introduce information that would only become available later and could cause information leakage.",

        # Reference answer for Question 8
        "Because the absence of these features indicated that the customer or part had never been sold before. This missingness could therefore directly reveal the target class and cause information leakage.",

        # Reference answer for Question 9
        "The validation results showed that the model performed best within a threshold range of approximately 0.55 to 0.65. A threshold of 0.62 was selected based on these results.",

        # Reference answer for Question 10
        "Traditional spare-parts forecasting generally predicts the quantity of future demand using historical sales data and time-series methods. This study instead predicts whether an individual RFQ will result in a sale, using supervised machine learning and information available when the RFQ is received."
    ]
}


dataset = Dataset.from_dict(data)

# for i, contexts in enumerate(data["contexts"], start=1):
#     text = "\n\n".join(contexts)

    # print("=" * 60)
    # print(f"Question {i}")
    # print(f"Number of chunks: {len(contexts)}")
    # print(f"Characters: {len(text):,}")
    # print(f"Words: {len(text.split()):,}")
