
#Load the dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes
import seaborn as sns


def load_dataset():
    valid_input = False
    while not valid_input:
        try:
            user_choice = input("Type '1' to choose the diabetes dataset or type '2' to choose the california dataset\n")
            user_choice = user_choice.lower()
            california = fetch_california_housing()  #home
            diabetes = load_diabetes() #sugar

            if user_choice == "2":
                print(california.DESCR)
                df_target = california.target
                df_data = california.data
                df_features = california.feature_names
                df = pd.DataFrame(california.data, 
                                         columns=california.feature_names)
                df['Target'] = pd.Series(california.target)
                valid_input = True
            elif user_choice == "1":
                print(diabetes.DESCR)
                df = pd.DataFrame(diabetes.data, 
                                         columns=diabetes.feature_names)
                df['Target'] = diabetes.target
                df_target = diabetes.target
                df_data = diabetes.data
                df_features = diabetes.feature_names
                valid_input = True
            else: 
                raise ValueError("Invalid dataset choice")
        except ValueError as e:
            print("Error: ", e)
            df = None
            df_target = None
            df_data = None
            df_features = None
        
    return df, df_target, df_data, df_features

def description(df_data, df_target, df_features, df):
    print("This is the current shape of the dataset:\n")
    print("Data Attributes:\n")
    print(df_data.shape)
    print("Data Targets:\n")
    print(df_target.shape)
    print("Data Features:\n")
    print(df_features)
    print("Data Precision:\n")
    pd.set_option('display.precision', 4)  # 4 digit precision for floats
    print("First 5 Rows:\n")
    print(df.head())
    print("Data Summary:\n")
    print(df.describe())



def split_data(df):
    from sklearn.model_selection import train_test_split
    X = df.drop("Target", axis=1)
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)
    
    return X_train, X_test, y_train, y_test



def training_model(X_train, y_train, df_features):
    from sklearn.linear_model import LinearRegression
    linear_regression = LinearRegression()
    linear_regression.fit(X=X_train, y=y_train)
    print("Linear Regression Coeficient:\n")
    for i, name in enumerate(df_features):
        print(f'{name:>10}: {linear_regression.coef_[i]}')  
        linear_regression.intercept_
    return linear_regression


def testing_model(X_test, linear_regression, y_test):
    from sklearn import metrics
    predicted = linear_regression.predict(X_test)
    expected = y_test
    predicted[:5]  # first 5 predictions
    expected[:5]   # first five targets 
    df_pred = pd.DataFrame()
    df_pred['Expected'] = pd.Series(expected)
    df_pred['Predicted'] = pd.Series(predicted)
    mae = metrics.mean_absolute_error(expected, predicted)
    print("\nMean absolute error:", mae,"\n")
    
    return predicted, df_pred, expected


def evaluate(df_pred, predicted,expected):

    figure = plt.figure(figsize=(9, 9))
    
    axes = sns.scatterplot(data=df_pred, x='Expected', y='Predicted', 
        hue='Predicted', palette='cool', legend=False)
    
    start = min(expected.min(), predicted.min())
    
    end = max(expected.max(), predicted.max())
    
    axes.set_xlim(start, end)
    
    axes.set_ylim(start, end)
    
    line = plt.plot([start, end], [start, end], 'k--') 
    plt.show()

def model_metrics(predicted, expected):
    from sklearn import metrics
    print(metrics.r2_score(expected, predicted))
    
    

continue_program = True

while continue_program:
    user_input = input("Welcome to Carolino's Python Project MENU? (press '1' to Start or anyother key to leave)\n")
    if user_input.lower() == "1":
        continue_program = True
        df, df_target, df_data, df_features = load_dataset()
        print("Dataset has been loaded \n")
        user_input = input("press '1' to Explore the data or anyother key to return to the Menu) \n")
        if user_input.lower() == "1":
            print("The Description of the Data is as follows: \n")
            description(df_data, df_target, df_features, df)
            user_input = input("press '1' to Split the Data or anyother key to return to the Menu) \n")
            if user_input.lower() == "1":
                X_train, X_test, y_train, y_test = split_data(df)
                print("The data has been split")
                user_input = input("press '1' to train the model anyother key to return to the Menu) \n")
                if user_input.lower() == "1":
                    linear_regression = training_model(X_train, y_train, df_features)
                    print("The Model has been trained \n")
                    user_input = input("press '1' test the model or anyother key to return to the Menu) \n")
                    if user_input.lower() == "1":
                        predicted, df_pred, expected = testing_model(X_test, linear_regression, y_test)
                        user_input = input("press '1' to evaluate the model or anyother key to return to the Menu) \n")
                        if user_input.lower() == "1":
                            evaluate(df_pred, predicted, expected)
                            user_input = input("press '1' check the model metrics or anyother key to return to the Menu) \n")
                            if user_input.lower() == "1":
                                model_metrics(predicted, expected)
                                print("Thats the metrics expected/predicted")
                                
        
    else:
        continue_program = False