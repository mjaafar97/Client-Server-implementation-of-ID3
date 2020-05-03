# -*- coding: utf-8 -*-
"""
Created on Fri May  1 03:11:25 2020

@author: Mariam
"""
#!/usr/bin/python           # This is client.py file
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
year_bins = [1918, 1930 , 1940 , 1950, 1960,  1970, 1980 , 1990 , 2000]
"""
                            Helper Functions
"""
# a function that reverse's the label encoding (retrieves the code of a specific value)
def getCode(col , val ): # retrieves the value of the instance to be detected 
    #works for tge genre 1 qne  genre 2 in movies and zip values for the states 
    ix = np.where( col== val)
    return ix[0][0]
def retrieve_zips(col):
    search = SearchEngine(simple_zipcode=True)
    states = []
    for zipc in users['ZipCode']:
        zipcode = search.by_zipcode(zipc)
        state  = zipcode.state
        states = np.append(states, state)
    return states

def retrieve_zip(ZipCode):
    search = SearchEngine(simple_zipcode=True)
    zipcode = search.by_zipcode(ZipCode)
    state  = zipcode.state
    return state

def getMoviegenres(movies):
    movies[['Genre_1','Genre_2','Genre_3','Genre_4','Genre_5','Genre_6']]=movies.Genres.str.split("|",expand=True)
    movies = movies.fillna('Unknown')
    return np.unique(movies['Genre_1'])

def getstates(users):
    #Retrieving the Zip Code Corresponding to Every state 
    states =  retrieve_zips(users['ZipCode'])
    users['states'] = states
    users = users.fillna('Unknown')
    states = np.unique(users['states'])
    return states


"""
                            Connection part 
"""
s = socket.socket()         # Create a socket object
host = socket.gethostname() # Get local machine name
port = 12345                # Reserve a port for your service.
s.connect((host, port))

"""
                        Take the user input
"""                
                #Taking Gender and processing 
gender= str(input("Please enter your Gender F for female and M for male :  "))
while (gender!= 'F'and gender!= 'M' ):
    gender  = str(input("Invalid please Re-Enter"))
if (gender == 'F'):
    gender  = 0 
else:
    gender = 1 
                #Taking Age and processing 
Age = int(input("Please Enter your age :  "))
if (Age >= Age_bins[3]):
    Age=4
elif (Age>= Age_bins[2]):
    Age=3
elif (Age >= Age_bins[1]):
    Age =2
else:
    Age =1 
             #Taking production year and processing 
Year  = int(input("Please enter the production year :  ")) 
if (Year >= year_bins[7]):
    Year=8
elif (Year >= year_bins[6]):
    Year=7
elif (Year >= year_bins[5]):
    Year=6
elif (Year >= year_bins[4]):
    Year=5
elif (Year >= year_bins[3]):
    Year=4
elif (Year >= year_bins[2]):
    Year=3
elif (Year >= year_bins[1]):
    Year=2
else:
    Year=1
            #Taking Movie genres and processing

print ('Choose the movie genre(s) : ')
genres = getMoviegenres(movies)
for i in range(len(genres)):
    print ("Choose   " + str(i) + "   for  " + str(genres[i]))
genre_1 = int(input('First Genre  :  '))
genre_2 = int(input('Second Genre  :  '))

            #Taking Zip Code genres and processing

ZipCode =  str(input("Please Enter your ZipCode :   "))
state = retrieve_zip(ZipCode)
states = getstates(users)
code = getCode(states , state)
state = code

            #Taking Occupation Code genres and processing

print("Please select your occupation")
print('\n')

print ( "0:  other or not specified")
print ("1:  academic/educator")
print ("2:  artist")
print ("4:  clerical/admin")
print  ("4:  college/grad student")
print ("5:  customer service")
print ("6:  doctor/health care")
print( "7:  executive/managerial")
print ("8:  farmer")
print ("9:  homemaker") 
print ("10:  K-12 student") 
print ( "11:  lawyer")
print ( "12:  programmer")
print ("13:  retired")
print ("14:  sales/marketing")
print ("15:  scientist")
print ("16:  self-employed")
print ("17:  technician/engineer")
print ("18:  tradesman/craftsman")
print ("19:  unemployed")
print ("20:  writer")
occupation= str ( input (" "))


msg = (str(Age)+ ','+ str(gender)+ ','+ str(genre_1) + ','+ str(genre_2)  +','+ str(Year)+ ',' + str(state) + ',' + str(occupation))
encoded = msg.encode()
s.sendall(encoded)


s.close()                     # Close the socket when done

