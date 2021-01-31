from lightgbm import LGBMClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import joblib
from azureml.core.run import Run

def main():     
    # Add arguments to script
    parser = argparse.ArgumentParser()

    # See lightgbm library for python for a list of parameters: https://lightgbm.readthedocs.io/en/latest/Parameters.html
    parser.add_argument('--n_estimators', type=int, default=100, help="number of boosting iterations")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="shrinkage rate")
    parser.add_argument('--max_depth', type=int, default=-1, help="max depth for tree model")
    parser.add_argument('--subsample', type=float, default=1.0, help="randomly select part of data without resampling. useful to speed up training and prevent over-fitting")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("n_estimators:", np.int(args.n_estimators))
    run.log("learning_rate:", np.float(args.learning_rate)) # see here for more ideas = https://bit.ly/3c2zJOm & https://bit.ly/3o6OAth
    run.log("max_depth:", np.int(args.max_depth))
    run.log("subsample:", np.float(args.subsample))
    
    # training set
    train_split_data = run.input_datasets["output_split_train"]
    # train_split_data = train_split_data.parse_parquet_files()
    train_split_df = train_split_data.to_pandas_dataframe()
    print(train_split_df.head(10))
    
    x_train = train_split_df.loc[:, train_split_df.columns != 'Exited']  
    y_train = train_split_df.loc[:, train_split_df.columns == 'Exited']  

    #evaluation set
    test_split_data = run.input_datasets["output_split_test"]
    test_split_df = test_split_data.to_pandas_dataframe()
    
    x_test = test_split_df.loc[:, test_split_df.columns != 'Exited']  
    y_test = test_split_df.loc[:, test_split_df.columns == 'Exited']  
    
    print(x_train.head(10))
    print(x_test.head(10))
    
    # declaring our model with parameters - default and those declared in our hyperparameter space
    model = LGBMClassifier(n_estimators=args.n_estimators, learning_rate=args.learning_rate, max_depth=args.max_depth, subsample=args.subsample).fit(x_train, y_train)

    # save model
    os.makedirs('./outputs/model', exist_ok=True)
    
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(model, './outputs/model/saved_model.joblib') 
    
    
    accuracy = model.score(x_test, y_test)
    print(model)
    print(x_test.head(10))
    
    run.log("Accuracy", np.float(accuracy)) #source: https://bit.ly/3mTxEWR && https://bit.ly/3hgonXx
  
    y_pred = model.predict(x_test)
    auc_weighted = roc_auc_score(y_pred, y_test, average='weighted')
    run.log("AUC_weighted", np.float(auc_weighted)) #source: https://bit.ly/3mTxEWR && https://bit.ly/3hgonXx
    
    # creating a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

if __name__ == '__main__':
    main()