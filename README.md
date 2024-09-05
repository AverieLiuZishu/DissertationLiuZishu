This file is an appendix for a dissertation presented to Durham University, Degree of Master of Data Science by Z0199448
To run the code, please unzip the ChiPiresults.zip first and then run the dissertation_column_error_solution.py, which is used for combining the data from ChiPi and fixing the messy columns data
dissertation_RF.py is the random forest model
dissertation_svm_normal.py is SVM model: As it takes huge time to train with huge dataset by SVM，I save these two SVM model as svm_model.pkl and svm_model2.pkl for convenience.
dissertation_xgb.py is XGBoost model
ChiPiresults.zip contains all the results data from ChiPi program
output_separated.zip contains the ChiPi result data without column error
output.zip is the output from dissertation_column_error_solution.py, as well as the combined file of all the seperated file in output_separated.zip
congloms.txt is the existing conglomerate list in 2019 I trying to reproduce and I covert it to list.csv for convenience
reproducedlist.csv is the output of the project than I trying to reproduce as accurate as possible with conglomerate list in 2019
