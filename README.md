# Breast Cancer Coimbra dataset SVM Classifier

This data was created to be able to identify a biomarker for breast cancer from a simple blood analysis. It has 116 samples each with 9 features classified into either having breast cancer or not.

I chose to implement using SVMs as they offered a better performance. The code is properly documented and commented so its easy to follow through and understand.

So far the best performance I have achieved is 88% using a gaussian Kernel and using only 5 features. The method I used for selecting the best features was sequential feature selection and had it rank all the features from best to worst. I then looped all through the features starting from 1 and then adding one more according to the rank on each iteration. In each iteraion I trained a model using the number of features equal to the iteration and then evaluated. This was done for upto 5 kernel functions. After that I chose the best performing Kernel and the respective group of best features to train my final model

# Hyperparameter Optimizations
For this I used the bayesian optimizer to help improve my model and boost my accuracy performance.

Feature selection prove very vaiuable as I was able to improve the SVM performance from around 70% to the current 88%
