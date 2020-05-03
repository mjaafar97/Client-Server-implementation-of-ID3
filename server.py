# -*- coding: utf-8 -*-
"""
Created on Fri May  1 03:11:11 2020

@author: Mariam
"""

#!/usr/bin/python           # This is server.py file
"""
Imports

"""
import socket               # Import socket module
import os
import numpy as np
import pandas as pd
from uszipcode import SearchEngine
os.chdir("C:\\Users\\Mariam") 
movies=pd.read_csv('movies.dat',sep ='::', names=['MovieID','Title','Genres'],skipinitialspace=True)
ratings=pd.read_csv('ratings.dat',sep ='::', names=['UserID','MovieID','Rating','Timestamp'],skipinitialspace=True)
users=pd.read_csv('users.dat',sep ='::', names=['UserID','Gender','Age','Occupation','ZipCode'],skipinitialspace=True)
Age_bins = [0, 13 , 19 , 45 , 56]
"""
Globals
"""
year_bins = [1918, 1930 , 1940 , 1950, 1960,  1970, 1980 , 1990 , 2000]
"""
Helper functions 

"""

# A function that does label encoding 
def encode(df,  col ):
    unique_vals = np.unique(df[col])
    for i in range(len(unique_vals)):
        df.loc[df[col] == unique_vals[i], col] = i 
    return df
# A function that does the binning for the data 
def binning (df , col , bins, bin_labels , encode = True):
    df[col]= pd.cut(df[col] , bins , labels =bin_labels  )
    return df
#converts column values into two values based on a specifich threshold 
def binarize(df , col , thresh , num = True):
    if (num):
        df.loc[df[col] <= thresh, col] = 0 
        df.loc[df[col] > thresh, col] = 1 
    else:
        df.loc[df[col] == thresh, col] = 0 
        df.loc[df[col] !=  0 , col] = 1 
    return df    

def retrieve_zips(col):
    search = SearchEngine(simple_zipcode=True)
    states = []
    for zipc in users['ZipCode']:
        zipcode = search.by_zipcode(zipc)
        state  = zipcode.state
        states = np.append(states, state)
    return states
def process_users(users):
    #Retrieving the Zip Code Corresponding to Every state 
    states =  retrieve_zips(users['ZipCode'])
    users['states'] = states 
    Users = users.drop('ZipCode', axis = 1 )
    Users = Users.fillna('Unknown')
    states = np.unique(Users['states'])
    Users = encode (Users, 'states')
    #Will devide the user age into 4 groups (child 1:12) teen (13:19) Adult (20:45) old (46:56)
    Age_bins = [0, 13 , 19 , 45 , 56]
    age_ranges = [1 , 2 , 3 , 4]
    Users= binning (Users , 'Age' , Age_bins, age_ranges , encode = True)
    Users =binarize(Users , 'Gender' , np.unique(Users['Gender'])[0], False)
    return  Users

"""
Data Cleaning Functions

"""




def process_Movies(movies):
    #split Movie genres 
    movies[['Genre_1','Genre_2','Genre_3','Genre_4','Genre_5','Genre_6']]=movies.Genres.str.split("|",expand=True)
    #pick top 2 genres 
    movies = movies.drop('Genres' , axis = 1)
    movies = movies.drop(['Genre_3','Genre_4','Genre_5','Genre_6'] , axis = 1)
    #retrieve movie year 
    movies['Year']=movies.Title.str.slice(start=-6)
    movies['Title']=movies.Title.str.slice(stop=-6)
    movies['Year']=movies.Year.str.slice(start=1)
    movies['Year']=movies.Year.str.slice(stop=-1)
    movies['Year']=pd.to_numeric(movies['Year'])
    #binning the movie year 
    Year_ranges = [1 , 2 , 3 , 4, 5 , 6 , 7 , 8]
    movies= binning(movies, 'Year' ,year_bins , Year_ranges )
    movies= movies.drop('Title', axis=1)
    #processing Movie genres 
    movies['Genre_2']= movies['Genre_2'].fillna('Unknown')
    movies['Genre_1']= movies['Genre_1'].fillna('Unknown')
    movies = encode(movies ,  'Genre_1' )
    movies = encode(movies ,  'Genre_2' ) 
    
    return movies
def process_ratings(ratings):
    ratings = binarize(ratings , 'Rating' , 3)
    ratings = ratings.drop('Timestamp', axis =1 )
    return ratings 

Mov = process_Movies(movies)
ratings = process_ratings(ratings)
Users = process_users(users)



Usermovies = pd.merge(ratings, Mov , on='MovieID', how='inner')
Usermovies = pd.merge(Usermovies, Users , on='UserID', how='inner')
Usermovies =  Usermovies.rename(columns={"Rating": "class"})
Usermovies = Usermovies.drop(['UserID','MovieID'] , axis=1)

def entropy(target_col):
    """
    Calculate the entropy of a dataset.
    The only parameter of this function is the target_col parameter which specifies the target column
    """
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

def InfoGain(data,split_attribute_name,target_name="class"):
    
    # entropy of y
    total_entropy = entropy(data[target_name])
    
    
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


def ID3(data,originaldata,features,target_attribute_name="class",parent_node_class = None):
    
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]
        
    elif len(features) ==0:
        return parent_node_class
    
    
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
        
        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features] #Return the information gain values for the features in the dataset
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
    
        tree = {best_feature:{}}
        
        
        features = [i for i in features if i != best_feature]
        
        
        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
            
            subtree = ID3(sub_data,Usermovies,features,target_attribute_name,parent_node_class)
            
            tree[best_feature][value] = subtree
            
        return(tree)    
        
def predict(query,tree,default = 1):    
    for key in list(query.keys()):
        if key in list(tree.keys()):
        
            try:
                result = tree[key][query[key]] 
            except:
                return default
  
            
            result = tree[key][query[key]]
            
            if isinstance(result,dict):
                return predict(query,result)

            else:
                return result

def test(data,tree):
    #Create new query instances by simply removing the target feature column from the original dataset and 
    #convert it to a dictionary
    queries = data.to_dict(orient = "records")
    
    #Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = []
    
    #Calculate the prediction accuracy
    for i in range(len(data)):
        p = predict(queries[i],tree,1.0) 
        predicted = np.append(predicted , p)
    return predicted

Y = Usermovies['class']
X = Usermovies.drop('class', axis=1) 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=42)

X_train['class'] = y_train

tree = ID3(X_train,X_train,X_train.columns[:-1])
print("Application is ready")
n_sample= 10000
x_train = X_train.drop('class' , axis =1 )
x_train = x_train[1:n_sample]
y = test (X_test , tree)
# SOCKET STUFF 
s = socket.socket()         # Create a socket object
host = socket.gethostname() # Get local machine name
port = 12345                # Reserve a port for your service.
s.bind((host, port))        # Bind to the port

s.listen(5)                 # Now wait for client connection.
while True:
   c, addr = s.accept()     # Establish connection with client.
   print ('Got connection from', addr)
   msg = c.recv(1024)
   msg = msg.decode()
   arr =msg.split(',', 7)
   Age=int(arr[0]) 
   gender= int(arr[1])
   genre_1 = int(arr[2])
   genre_2 = int(arr[3])
   Year= int(arr[4]) 
   state= int(arr[5])
   occupation =int( arr[6])
   x = {
      'Genre_1': genre_1,
      'Genre_2': genre_2,
      'Year': Year,
      'Gender': gender ,
      'Age': Age,
      'Occupation': occupation ,
      'states': state}
   p = predict(x,tree,1.0) 
   if (p>0):
       print("Go Ahead you will like this movie!")
   else:
       print("Perhaps you would like to try another movie")
   c.close()                # Close the connectio