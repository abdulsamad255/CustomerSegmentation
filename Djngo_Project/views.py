from django.shortcuts import render
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

def user(request):
    return render(request, 'userinput.html')

def viewdata(request):
    # Load the datasets
    df1 = pd.read_csv('C:/Users/PMLS/Documents/ML/ML Algorithms/Unsuppervised Machine Learning Project/Train.csv')
    df2 = pd.read_csv('C:/Users/PMLS/Documents/ML/ML Algorithms/Unsuppervised Machine Learning Project/Test.csv')
    
    # Drop duplicates
    df2 = df2.drop_duplicates(keep='last')
    
    # Concatenate datasets
    combine_data = pd.concat([df1, df2], axis=0, ignore_index=True)
    
    # Drop duplicate rows
    combine_data = combine_data.drop_duplicates(keep='last')
    
    # Drop columns with a high percentage of missing values
    num_miss_vars = combine_data.select_dtypes(["int64", "float64"]).columns
    missing_percentage = combine_data[num_miss_vars].isnull().mean() * 100
    cols_to_drop = missing_percentage[missing_percentage > 32].index
    combine_data = combine_data.drop(columns='ID')
    
    # Fill missing values
    numerical_columns = ['Work_Experience', 'Family_Size']
    for col in numerical_columns:
        if combine_data[col].isnull().sum() > 0:
            combine_data[col].fillna(combine_data[col].mean(), inplace=True)
    
    # Prepare data for KMeans
    X = combine_data.drop(columns='Segmentation', axis=1)
    y = combine_data['Segmentation']
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col])
    
    # Train KMeans model
    kmeans = KMeans(n_clusters=3, init='k-means++')
    kmeans.fit(X)
    
    # Retrieve input data from GET request
    new_data = {
        'Gender': request.GET['Gender'],
        'Ever_Married': request.GET['Ever_Married'],
        'Age': float(request.GET['Age']),
        'Graduated': request.GET['Graduated'],
        'Profession': request.GET['Profession'],
        'Work_Experience': float(request.GET['Work_Experience']),
        'Spending_Score': request.GET['Spending_Score'],
        'Family_Size': float(request.GET['Family_Size']),
        'Var_1': request.GET['Var_1']
    }
    
    new_data_df = pd.DataFrame([new_data])
    
    # Encode categorical variables for new data
    for col in new_data_df.select_dtypes(include=['object']).columns:
        new_data_df[col] = le.fit_transform(new_data_df[col])
    
    # Predict cluster
    cluster = kmeans.predict(new_data_df)
    segment = cluster[0]  # Access the single cluster prediction directly
    
    # Define segment descriptions
    messages = {
        0: 'Lowest engagement Customer: Customers with minimal interaction or spending.',
        1: 'High Engagement Customer: Customers who frequently interact and spend a lot.',
        2: 'Moderate Engagement Customer: Customers who have a balanced level of interaction and spending.',
        3: 'Low-value or infrequent buyers; may need incentives to increase engagement.'
        
    }
    
    # Prepare the data to render
    data = {
        'prediction': segment,
        'message': messages.get(segment)
    }
    
    return render(request, 'viewdata.html', data)
