# Breast Cancer Coimbra dataset SVM Classifier
The code prepares and processes the dataset from a csv file, it performs feature analysis and ranking of features then optimizes the SVM model using a bayesian optimizer. The model is evaluated using K-fold cross-validation, theres is no hold out validation performed. A confusion matrix is also generated to produce the recall,precision and F1 score of the model.

The code is properly structured, well commented and symbolic variables used allowing beginners too have an easy time understanding it.

Please also do cite my work if you like it :-)

# Dataset Brief
The dataset was obtained from the UCI Machine Learning Repisotory : Breast Cancer Coimbra Data Set 
It features 116 samples each with 9 attributes or features:
Age (years) 
BMI (kg/m2)
Glucose (mg/dL) 
Insulin (µU/mL) 
HOMA
Leptin (ng/mL) 
Adiponectin (µg/mL) 
Resistin (ng/mL) 
MCP-1(pg/dL) 

It features only 2 classes
1=Healthy controls 
2=Patients

Its a new attempt at early detection of breast cancer using a simple blood analysis other than commonly used mammogram scans
More information on the dataset https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra
The highest accuracy so far achieved has been 88% after exclusion of two of the features Adiponectin and MCP.

# Methodology
So far the best performance I have achieved is 88% using a gaussian Kernel and using only 5 features. The method I used for selecting the best features was sequential feature selection and had it rank all the features from best to worst. I then looped all through the features starting from 1 and then adding one more according to the rank on each iteration. In each iteraion I trained a model using the number of features equal to the iteration and then evaluated. This was done for upto 5 kernel functions. After that I chose the best performing Kernel and the respective group of best features to train my final model.
