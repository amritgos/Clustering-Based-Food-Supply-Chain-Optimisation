#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 19:17:10 2018

@author: amritgos
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
import googlemaps


train = "/home/amritgos/Documents/Amrit/Study/Project/HK Paper2/Codes/Villages.csv" 
df = pd.read_csv(train)

gmaps_key = googlemaps.Client(key = "AIzaSyB3zniJdTV5EViYp5w1gS5O9iwBKLNthts")

df["lat"] = None
df["lon"] = None

print(df.head())
for i in range(len(df)):
    print("hi")
    geocode_result = gmaps_key.geocode(df.iat[i,0])
    print("hey")
    try :
        lat = geocode_result[0]["geometry"]["location"]["lat"]
        lon = geocode_result[0]["geometry"]["location"]["lon"]
        
        df.iat[i, df.columns.get_loc("lat")] = lat
        df.iat[i, df.columns.get_loc("lon")] = lon
        
        print(df[i])
    except:
        lat = None
        lon = None
print(df.head())