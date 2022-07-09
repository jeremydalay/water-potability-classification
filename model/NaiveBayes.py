import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def naive_bayes_model(dataset, test_size, random):

    # Defining the features and the target variables
    features = []
    target = ""

    # Create a train, test split. 
    features_train, features_test, target_train, target_test = train_test_split(dataset[features],
                                                                                dataset[target], 
                                                                                test_size = test_size,
                                                                                random_state = random)
    
    model = GaussianNB()
    model.fit(features_train, target_train)

    # After fitting, we will make predictions using the testing dataset
    pred = model.predict(features_test)
    accuracy = accuracy_score(target_test, pred)

    # Displaying the accuracy of the model
    print("Model Accuracy = ", accuracy*100,"%")

    # answer = model.predict([[2,2,1,0]]) # sunny, mild, normal, false

    # if answer == 1:
    #     print("Play")
    # elif answer == 0:
    #     print("No Play")
