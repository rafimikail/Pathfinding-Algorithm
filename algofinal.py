import import_ipynb
import pandas, os, geopy 
from pandas import DataFrame

from geopy.distance import geodesic, lonlat, distance

#PROBLEM 1 --------------------------------------------------------------------------------------------------------

print("--------------------------------- PROBLEM 1 ---------------------------------")

df1 = pandas.read_csv("locationsofstops.csv")
df2 = pandas.read_csv("distancebetweenstops.csv")

from geopy.geocoders import Nominatim
nom = geopy.geocoders.Nominatim(user_agent="my app")
df1["Address"] = df1["location"].apply(nom.geocode, timeout = 10)

#Task 1 : Marking and getting stops 

df1["Latitude"] = df1["Address"].apply(lambda x: x.latitude if x != None else None)
df1["Longitude"] = df1["Address"].apply(lambda x: x.longitude if x != None else None)
df1["Coordinate"] = df1["Latitude"].astype(str)+","+df1["Longitude"].astype(str)

#Task 2 : Adding distance between stops
#STOP 1
#KL Sentral -KLIA
origin = (df1.at[0,"Coordinate"])
stop = (df1.at[1,"Coordinate"])
distance1 = int(geodesic(origin,stop).km)

#KL Sentral - Terminal Skypark Komuter Station
origin1 = (df1.at[0,"Coordinate"])
stop1 = (df1.at[9,"Coordinate"])
distance12 = int(geodesic(origin1, stop1).km)

df2['Stop 1'].replace({"Kuala Lumpur International Airport":"Kuala Lumpur International Airport"+", "
                       +str(distance1)+" km"}, inplace = True)
df2['Stop 1'].replace({"Terminal Skypark Komuter Station":"Terminal Skypark Komuter Station"+", "
                       +str(distance12)+" km"}, inplace = True)

#STOP 2
#KLIA - Terminal TUTA
origin2 = (df1.at[1,"Coordinate"])
stop2 = (df1.at[2,"Coordinate"])
distance2 = int(geodesic(origin2, stop2).km)

#Terminal Skypark Komuter - Sultan Abdul Aziz Shah Airport
origin3 = (df1.at[9,"Coordinate"])
stop3 = (df1.at[10,"Coordinate"])
distance13 = int(geodesic(origin3, stop3).km)

#KLIA - Changi International Airport
origin4 = (df1.at[1,"Coordinate"])
stop4 = (df1.at[4,"Coordinate"])
distance4 = int(geodesic(origin4, stop4).km)

#KLIA - Senai Airport
origin5 = (df1.at[1,"Coordinate"])
stop5 = (df1.at[7,"Coordinate"])
distance8 = int(geodesic(origin5, stop5).km)

#Adding distance
df2['Stop 2'].replace({"Terminal TUTA":"Terminal TUTA"+", "+str(distance2)+" km"}, inplace = True)
df2['Stop 2'].replace({"Sultan Abdul Aziz Shah Airport":"Sultan Abdul Aziz Shah Airport"+", "
                       +str(distance13)+" km"}, inplace = True)
df2['Stop 2'].replace({"Changi Airport":"Changi Airport"+", "+str(distance4)+" km"}, inplace = True)
df2['Stop 2'].replace({"Senai Airport":"Senai Airport"+", "+str(distance8)+" km"}, inplace = True)

#Stop 3
#Terminal TUTA - Legoland
origin6 = (df1.at[2,"Coordinate"])
stop6 = (df1.at[3,"Coordinate"])
distance3 = int(geodesic(origin6, stop6).km)

#Sultan Abdul Aziz Shah Airport - Senai Airport
origin7 = (df1.at[10,"Coordinate"])
stop7 = (df1.at[7,"Coordinate"])
distance11 = int(geodesic(origin7, stop7).km)

#Sultan Abdul Aziz Shah Airport - Singapore Seletar Airport
origin8 = (df1.at[10,"Coordinate"])
stop8 = (df1.at[11,"Coordinate"])
distance14 = int(geodesic(origin8, stop8).km)

#Changi Airport - Tanah Merah MRT Station
origin9 = (df1.at[4,"Coordinate"])
stop9 = (df1.at[5,"Coordinate"])
distance5 = int(geodesic(origin9, stop9).km)

#Senai Airport - Johor Bahru Sentral
origin10 = (df1.at[7,"Coordinate"])
stop10 = (df1.at[8,"Coordinate"])
distance9 = int(geodesic(origin10, stop10).km)

#Adding Distance
df2['Stop 3'].replace({"Legoland":"Legoland"+", "+str(distance3)+" km"}, inplace = True)
df2['Stop 3'].replace({"Senai Airport":"Senai Airport"+", "+str(distance11)+" km"}, inplace = True)
df2['Stop 3'].replace({"Singapore Seletar Airport":"Singapore Seletar Airport"+", "+str(distance14)+" km"}, inplace = True)
df2['Stop 3'].replace({"Tanah Merah MRT Station":"Tanah Merah MRT Station"+", "+str(distance5)+" km"}, inplace = True)
df2['Stop 3'].replace({"Johor Bahru Sentral":"Johor Bahru Sentral"+", "+str(distance9)+" km"}, inplace = True)

#Stop 4
#Senai Airport - Johor Bahru Sentral
origin11 = (df1.at[7,"Coordinate"])
stop11 = (df1.at[8,"Coordinate"])
distance9 = int(geodesic(origin11, stop11).km)

#Singapore Seletar Airport - Legoland
origin12 = (df1.at[11,"Coordinate"])
stop12 = (df1.at[3,"Coordinate"])
distance15 = int(geodesic(origin12, stop12).km)

#Tanah Merah MRT Station - Jurong East Station
origin13 = (df1.at[5,"Coordinate"])
stop13 = (df1.at[6,"Coordinate"])
distance6 = int(geodesic(origin13, stop13).km)

#Johor Bahru Sentral - Legoland
origin14 = (df1.at[8,"Coordinate"])
stop14 = (df1.at[3,"Coordinate"])
distance10 = int(geodesic(origin14, stop14).km)


#can but no change
df2['Stop 4'].replace({'Johor Bahru Sentral':'Johor Bahru Sentral'+',' +str(distance9)+" km"}, inplace = True)
df2['Stop 4'].replace({"Legoland":"Legoland"+", "+str(distance15)+" km"}, inplace = True)
df2['Stop 4'].replace({"Jurong East Station":"Jurong East Station"+", "+str(distance6)+" km"}, inplace = True)
df2['Stop 4'].replace({" Legoland":"Legoland"+", "+str(distance10)+" km"}, inplace = True)


#Stop 5
#Johor Bahru Sentral - Legoland
origin15 = (df1.at[8,"Coordinate"])
stop15 = (df1.at[3,"Coordinate"])
distance10 = int(geodesic(origin15, stop15).km)

#Jurong East Station - Legoland
origin16 = (df1.at[6,"Coordinate"])
stop16 = (df1.at[3,"Coordinate"])
distance7 = int(geodesic(origin16, stop16).km)


#can
#df2['Stop 5'].replace({"Nan":"Nan"},inplace = True)
df2['Stop 5'].replace({"Legoland":"Legoland"+", "+str(distance10)+" km"}, inplace = True)
#df2['Stop 5'].replace({"Nan":"Nan"}, inplace = True)
df2['Stop 5'].replace({" Legoland":"Legoland"+", "+str(distance7)+" km"}, inplace = True)
#df2['Stop 5'].replace({"Nan":"Nan"}, inplace = True)


print(df1)

print(df2)

  
#TASK 3 : FIND BEST ROUTE USING DIJKSTRA ALGORITHM
graph = {'KL Sentral':{'KLIA':42,'KTM Skypark':14},'KLIA':{'TUTA':256,'Changi':247,'Senai':250},
         'TUTA':{'Legoland':10},'Changi':{'MRT Tanah Merah':5},'MRT Tanah Merah':{'Jurong':26}
         ,'Jurong':{'Legoland':15},'Senai':{'JBS':22},'JBS':{'Legoland':15},
         'KTM Skypark':{'Subang Airport':0},'Subang Airport':{'Senai':287,'Seletar':314},
         'Seletar':{'Legoland':26},'Legoland':{'Seletar':26}}

def dijkstra(graph,start,goal):
    shortest_distance = {}
    predecessor = {}
    unseenNodes = graph
    infinity = 9999999
    path = []
    for node in unseenNodes:
        shortest_distance[node] = infinity
    shortest_distance[start] = 0
 
    while unseenNodes:
        minNode = None
        for node in unseenNodes:
            if minNode is None:
                minNode = node
            elif shortest_distance[node] < shortest_distance[minNode]:
                minNode = node
 
        for childNode, weight in graph[minNode].items():
            if weight + shortest_distance[minNode] < shortest_distance[childNode]:
                shortest_distance[childNode] = weight + shortest_distance[minNode]
                predecessor[childNode] = minNode
        unseenNodes.pop(minNode)
 
    currentNode = goal
    while currentNode != start:
        try:
            path.insert(0,currentNode)
            currentNode = predecessor[currentNode]
        except KeyError:
            print('Path not reachable')
            break
    path.insert(0,start)
    if shortest_distance[goal] != infinity:
        print('Shortest distance is ' + str(shortest_distance[goal]))
        print('And the path is ' + str(path))

print(dijkstra(graph, "KL Sentral", "Legoland"))

#TASK 4 : PLOTTING ROUTE

import gmaps
#import googlemaps
from gmplot import gmplot
#gmaps = googlemaps.Client(api_key)
gmap = gmplot.GoogleMapPlotter(2.64, 102.803, 7)

gmap.scatter(df1['Latitude'],df1['Longitude'], '#FFFFFF', size=100, marker=False)
gmap.draw('maptest.html')

lat = df1["Latitude"].tolist()
lng = df1["Longitude"].tolist()

route0_lat = [lat[0],lat[1],lat[2],lat[3]]
route0_lng = [lng[0],lng[1],lng[2],lng[3]]
gmap.plot(route0_lat, route0_lng, "white", edge_width = 3.0)

route1_lat = [lat[0],lat[9],lat[10],lat[7],lat[8],lat[3]]
route1_lng = [lng[0],lng[9],lng[10],lng[7],lng[8],lng[3]]
gmap.plot(route1_lat, route1_lng, "blue", edge_width = 3.0)

route2_lat = [lat[0],lat[9],lat[10],lat[11],lat[3]]
route2_lng = [lng[0],lng[9],lng[10],lng[11],lng[3]]
gmap.plot(route2_lat, route2_lng, "green", edge_width = 3.0)

route3_lat = [lat[0],lat[1],lat[4],lat[5],lat[6],lat[3]]
route3_lng = [lng[0],lng[1],lng[4],lng[5],lng[6],lng[3]]
gmap.plot(route3_lat, route3_lng, "red", edge_width = 3.0)

route4_lat = [lat[0],lat[1],lat[7],lat[8],lat[3]]
route4_lng = [lng[0],lng[1],lng[7],lng[8],lng[3]]
gmap.plot(route4_lat, route4_lng, "yellow", edge_width = 3.0)


gmap.draw('Polyline.html')

print("Open file Polyline.html to see the illustration of the routes")


#PROBLEM 2 -------------------------------------------------------------------------------------------------------

print("--------------------------------- PROBLEM 2 ---------------------------------")

import string

import plotly
import plotly.graph_objects as go

filesArr = ["bus1.txt", "plane1.txt", "kliaexpress1.txt", "taxi1.txt", "ktm1.txt", "rapidkl1.txt", "smrt1.txt"]
posArr = ["posbus.txt", "posplane.txt", "posklia.txt", "postaxi.txt", "posktm.txt", "posrapid.txt", "posmrt.txt"]  # store stopwords for each text file
stopWords = "stopwords.txt"
wordCount_before = [0, 0, 0, 0, 0, 0, 0]
wordCount_after = [0, 0, 0, 0, 0, 0, 0]


# Method for remove stop words from each text file
def removeStopwords(list1, stopwords):
    return [w for w in list1 if w not in stopwords]


print("\nSTOP WORDS REMOVAL:\n")

index = 0
for i in range(0, len(filesArr), 1):
    fileDirectory = filesArr[index]
    posDirectory = posArr[index]

    # Count total words before stop words removal
    totalCount = 0
    with open(fileDirectory, "r+", encoding="utf8") as word_list:
        words = word_list.read()

        totalCount = len(words)

        print("Total word count of {0} before stop words removal: {1}".format(
            filesArr[index], totalCount))
        wordCount_before[index] = totalCount

        # file with stopwords
        f1 = open(stopWords, "r+", encoding="utf8")

        # transport text file
        f2 = open(fileDirectory, "r+", encoding="utf8")

        file1_raw = f1.read()
        file2_raw = f2.read().lower()

        sw = file1_raw.split()
        file2_words_SWRemoved = file2_raw.split()

        # Remove punctuations
        pc = string.punctuation
        pc += '""''-'
        table = str.maketrans('', '', pc)
        stripped = [w.translate(table) for w in file2_words_SWRemoved]

        # Remove stop words
        wordlist2 = removeStopwords(stripped, sw)

        # Write edited text file content back
        f2_w = open(posDirectory, "w", encoding="utf8")
        f2_w.write(" ".join(wordlist2))
        f2_w.close()
        f1.close()
        f2.close()
        #print(f)

        # Count total words in each transportation stop word file
        totalCount1 = 0
        with open(posDirectory, "r+", encoding="utf8") as list:
            words1 = list.read().lower().split()

        totalCount1 = len(words1)

        print("Total word count of {0} after stop words removal: {1}".format(
            filesArr[index], totalCount1))
        print(wordlist2)
        print("\n")
        wordCount_after[index] = totalCount1
        index = index + 1

# PROBLEM 2.6: Plot line/scatter/histogram using Plotly (Word Count, stop words)

x = ["Bus", "Plane", "KLIA Express", "Taxi", "KTM", "Rapid KL", "SMRT"]
z = [str(wordCount_after[0]), str(wordCount_after[1]), str(wordCount_after[2]), str(wordCount_after[3]),
     str(wordCount_after[4]), str(wordCount_after[5]), str(wordCount_after[6])]
y = [str(wordCount_before[0]), str(wordCount_before[1]), str(wordCount_before[2]), str(wordCount_before[3]),
     str(wordCount_before[4]), str(wordCount_before[5]), str(wordCount_before[6])]

fig = go.Figure()
fig.add_trace(go.Histogram(histfunc="sum", y=y, x=x, name="Word count"))
fig.add_trace(go.Histogram(histfunc="sum", y=z, x=x, name="Stop words"))

plotly.offline.plot(fig)

# PROBLEM 2.7: Negative/Positive words in article
transport = ["Bus", "Plane", "KLIA Express", "Taxi", "KTM", "Rapid KL", "SMRT"]
filesArr1 = ["bus1.txt", "plane1.txt", "kliaexpress1.txt", "taxi1.txt", "ktm1.txt", "rapidkl1.txt", "smrt1.txt"]

positiveCountArr = [0, 0, 0, 0, 0, 0, 0]
negativeCountArr = [0, 0, 0, 0, 0, 0, 0]
sentimentCountArr = [0, 0, 0, 0, 0, 0, 0]


# Compare method

def positiveCompare(file, posfile):
    return [w for w in file if w in posfile]


def negativeCompare(file, negfile):
    return [w for w in file if w in negfile]


print("-------------------POS/NEG COUNTER :------------------------\n")

index = 0
for i in range(0, len(filesArr), 1):
    fileDirect = filesArr1[index]

    # Count total words
    totalCount = 0
    with open(fileDirect, "r+", encoding="utf8") as word_list:
        words = word_list.read()

        totalCount = len(words)

        pos = open("positive.txt", encoding="utf8")
        pos = pos.read().split()
        neg = open("negative.txt", encoding="utf8")
        neg = neg.read().split()

        file1 = open(fileDirect, encoding='utf8')
        file1 = file1.read().lower().split()

        positiveCount = 0
        negativeCount = 0
        neutral = 0

        poswordlist = positiveCompare(file1, pos)
        positiveCount = len(poswordlist)

        negwordlist = negativeCompare(file1, neg)
        negativeCount = len(negwordlist)

        neutral = totalCount - (positiveCount - negativeCount)
        sentiment = (positiveCount - negativeCount)/100

        print("Total Number of Positive Words in {0}: {1}".format(filesArr1[index], positiveCount))
        print("Total Number of Negative Words in {0}: {1}".format(filesArr1[index], negativeCount))
        print("Total Number of Neutral Words in {0}: {1}".format(filesArr1[index], neutral))
        print("Sentiment Value in {0}: {1}".format(filesArr1[index], sentiment))
        # Conclusion made
        if positiveCount > negativeCount:
            print("Article Sentiment: Positive\n")
        elif positiveCount < negativeCount:
            print("Article Sentiment: Negative\n")
        else:
            print("Article Sentiment: Neutral\n")


        positiveCountArr[index] = positiveCount
        negativeCountArr[index] = negativeCount
        sentimentCountArr[index] = sentiment

        index = index + 1
        # sort the transportation according to the sentiment value
sentimentArr = {transport[0]: sentimentCountArr[0], transport[1]: sentimentCountArr[1],
                transport[2]: sentimentCountArr[2], transport[3]: sentimentCountArr[3],
                transport[4]: sentimentCountArr[4], transport[5]: sentimentCountArr[5],
                transport[6]: sentimentCountArr[6]}
sentimentArr = sorted(sentimentArr.items(), key=lambda x: x[1], reverse=True)

print('----------------Best Transportation According to Sentiment Value-----------------------')
for i in sentimentArr:
    print(i[0], i[1])

    # GRAPH FOR POSITIVE AND NEGATIVE WORDS

x = ["Bus", "Plane", "KLIA Express", "Taxi", "KTM", "Rapid KL", "SMRT"]
y = [str(positiveCountArr[0]), str(positiveCountArr[1]), str(positiveCountArr[2]), str(positiveCountArr[3]),
     str(positiveCountArr[4]), str(positiveCountArr[5]), str(positiveCountArr[6])]
z = [str(negativeCountArr[0]), str(negativeCountArr[1]), str(negativeCountArr[2]), str(negativeCountArr[3]),
     str(negativeCountArr[4]), str(negativeCountArr[5]), str(negativeCountArr[6])]

fig1 = go.Figure()
fig1.add_trace(go.Histogram(histfunc="sum", y=y, x=x, name="Positive words"))
fig1.add_trace(go.Histogram(histfunc="sum", y=z, x=x, name="Negative words"))

plotly.offline.plot(fig1)


#PROBLEM 3 ------------------------------------------------------------------------------------------

print("--------------------------------- PROBLEM 3 ---------------------------------")

# pos = {#number of positive
#     "KLIA EXPRESS": 9,
#     "SMRT": 15,
#     "KTM": 10,
#     "BUS": 9,
#     "PLANE": 38,
#     "TAXI": 27,
# }
#
# neg = {#number of negative
#     "KLIA EXPRESS": 8,
#     "SMRT": 7,
#     "KTM": 16,
#     "BUS": 12,
#     "PLANE": 18,
#     "TAXI":28,
# }

routeAtt = {
    "route":[["KL SENTRAL  > KLIA > TERMINAL TUTA > LEGOLAND"],
            ["KL SENTRAL  > TERMINAL SKYPARK KOMUTER STATION > SZB AIRPORT > SENAI AIRPORT > JB SENTRAL  > LEGOLAND"],
            ["KL SENTRAL  > TERMINAL SKYPARK KOMUTER STATION > SZB AIRPORT > SINGAPORE SELETAR > LEGOLAND"],
            ["KL SENTRAL  > KLIA > CHANGI AIRPORT > TANAH MERAH >JURONG EAST > LEGOLAND"],
            ["KL SENTRAL  > KLIA > SENAI AIRPORT > JB SENTRAL > LEGOLAND"]],

    "mode": [["KLIA EXPRESS", "BUS", "BUS"],
             ["KTM", "TAXI", "PLANE","BUS","BUS"],
             ["KTM", "TAXI", "PLANE","TAXI"],
             ["KLIA EXPRESS", "PLANE","SMRT","SMRT","BUS"],
             ["KLIA EXPRESS", "PLANE", "BUS", "BUS"]],
    "distance": [42 + 256 + 10,14 +0 + 287 + 22 + 15,14 + 0 + 319 + 26,42 + 297 + 5 + 22 + 15,42 + 250 + 22 + 15],
    "positive": [0,0,0,0,0],
    "negative": [0,0,0,0,0],
    "score": [0,0,0,0,0]
    }

# positive counter
i = 0;
for x in routeAtt["positive"]:
    total = 0;
    for y in routeAtt["mode"][i]:
        if y == "KLIA EXPRESS":
            total += 9
        elif y == "SMRT":
            total += 15
        elif y == "KTM":
            total += 10
        elif y == "BUS":
            total += 9
        elif y == "PLANE":
            total += 38
        else:   #taxi
            total += 27
    routeAtt["positive"][i]=total;
    i += 1;
#negative counter
i = 0;
for x in routeAtt["negative"]:
    total = 0;
    for y in routeAtt["mode"][i]:
        if y == "KLIA EXPRESS":
            total += 8
        elif y == "SMRT":
            total += 7
        elif y == "KTM":
            total += 16
        elif y == "BUS":
            total += 12
        elif y == "PLANE":
            total += 18
        else:  # taxi
            total += 28
    routeAtt["negative"][i]=total;
    i += 1
#score counter
i=0;
distanceWeightage = 50
negativeWeightage = 25
positiveWeightage = 25

for x in routeAtt["score"]:
    score= (routeAtt["distance"][i] * distanceWeightage) + (routeAtt["negative"][i] * negativeWeightage) - (routeAtt["positive"][i] * positiveWeightage);
    routeAtt["score"][i]=int(score)
    i+=1;

print("positive : ",routeAtt["positive"])
print("negative : ",routeAtt["negative"])
print("score : " ,routeAtt["score"])
print()

for i in range (len(routeAtt["route"])):
    print("Route ", i + 1," : ", routeAtt["route"][i])
    print("Transport: ", routeAtt["mode"][i])
    print("Positive: ", routeAtt["positive"][i])
    print("Negative: ", routeAtt["negative"][i])
    print("Score: ", routeAtt["score"][i])
    print()

#sorting
routeTemp = routeAtt["route"]
scoreTemp = routeAtt["score"]


import numpy
routeTemp = numpy.array(routeTemp)
scoreTemp = numpy.array(scoreTemp)
inds = scoreTemp.argsort()
pathSorted = routeTemp[inds]

#printing
print()
print("Recommended routes (1=Most recommended, 5=Least recommended)")
i = 1
for x in pathSorted:
    print(i, end = ": ")
    i+=1
    print(x)