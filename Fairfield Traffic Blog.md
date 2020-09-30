# Fairfield Conneticut Traffic Analysis: Folium, RFC & WordCloud
## Understanding the Relationship and Causes of Traffic in Fairfield CT 
![](https://i.imgur.com/m2mbg0r.jpg)
*Business Problem:*
   Automotive collisions have become a ubiquitous aspect of modern life. These accidents claim the lives and livelihoods of millions of people a year all over the world.  A study by the Department of Transportation's, [National Highway Traffic Saftey Administration](https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/812013) (NHTSA), placed a price tag of over a quarter-trillion dollars in 2010.  I have seen several projects diving into this problem from the perspective of loss of life property or employment, all of which are noble in nature. I want to explore something else, something more trivial, and infuriating TRAFFIC! That's right, and I will be analyzing and building a machine learning model to predict an accident's impact on traffic. Through analysis of observations like weather, time of year, location, and many more, we might be able to alleviate the modern leech on our precious free time.
   
   By better understanding the critical factors influencing a collision effect on traffic flow, we can reduce the number of total crashes and limit the time they steal from us. If done correctly, this work could help add precious time to our days.  To defeat them, we first understand them as Sun Tzu said, "know your opponent, and you will never lose'. We must learn what makes traffic tick, and we will do it through careful feature engineering and model choice. If we can establish a pattern,  we would deploy available resources to combat these conditions that worsen traffic. 

This analysis will help inform the driver of critical areas better and plan their route accordingly and decision-makers on the local level and above on resource allocation. 
   
   *TLDR* Car crashes have many issues associated with them that are horrible and life-changing. Traffic, however, affects us the most regularly. This analysis will aim to understand better the conditions that make a crash cause a higher traffic load.
   
   
## THE DATA
This US-Accident Dataset is a countrywide dataset covering 49 US states. It has 3.5 million separate traffic accidents taking place from February 2016 to December 2019. Based on definition of our problem, factors that will influence our decission are:

Information describing the collision
Various measurements and observations of the weather
Types of roadway or infrastructure involved We choose to use the data's entirety in our predictive efforts to better our target audience of governmental decision-makers. Remember, severity is only relational to the impact an accident has on traffic flow and will be our target variable in our later analysis.
Acknowledgments:
Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. "A Countrywide Traffic Accident Dataset.", 2019. Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. "Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights." In Proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019.

## THE METHODOLOGY
I wrote all the code on Kaggle's platform, then downloaded and pushed it to this Github repository. The data was derived from a Kaggle dataset by the name of [*US Accidents (3.5 million records)*](https://www.kaggle.com/sobhanmoosavi/us-accidents) as well as a small set of png files for word cloud masking. 

While the data set encompassed the 48 contiguous States that make up the US's main body, Connecticut's Fairfield County is at the center of this analysis. Fairfield is located in the Southwestern corner of the State and is the most populous and fast-growing county.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Map_of_Connecticut_highlighting_Fairfield_County.svg/600px-Map_of_Connecticut_highlighting_Fairfield_County.svg.png)

After defining our location, I corrected the boolean values into integer form and removed unnecessary ones. I dismissed 13 observations for lack of information or repeated information. I used a missing value visualization to evaluate the completeness of the data set. 

![](https://i.imgur.com/x0Rn0zT.png)

The yellow marks represent missing values. We see there are 13 columns with missing values. We will outright remove columns like he 'End_Lat/Lng' and 'Number' column as they are not useful. Then some will need imputation work like 'Wind_Speed(mph),' 'Wind_Chill(F),' and others. Some of these like 'TMC,' 'Weather_Conditions,' will require a closer examination.

Mean value imputation was utilized to fill in values for 'Wind_Cill(F)', 'Wind_Speed(mph)', 'Visibility(mi)', 'Precipitation(in)', 'Temperature(F)', 'Humidity(%)', and 'Pressure(in)'. To limit our random forest's depth and breadth, I binned or Bucketed out 'Times of Day' into Early Morning, Morning, Afternoon, and Evening. Months became seasons, and 'Day of the Week' was separated into Weekday and Weekend. Below are faceted grouped bar graphs for each bucketed category and the unbucketed data used to create it. We can see both the binned or bucketed data and its raw counterpart. On the left, we have two Seaborn Countplots that show us the relation of commuting and car accident frequency and severity. We can see that the afternoon seems to be a more difficult time of day. Understandably we can expect drivers to be more tired and thus more impaired after a full day of work. One interesting takeaway from the plots on the right is that Autumn seems to be a more dangerous time to be on the road. We found this interesting because our gut would have said that New England winters would have caused a more considerable uptick in accidents than the fall weather. Possibly due to holidays involving travel and the influx of 'Leafers' people who drive through CT to get to the more northern states to see the leaf change color.

![](https://i.imgur.com/vLTBl3x.png)

I'm always a sucker for good map visualization, so I was eager to employ *Folium* for this project. *Folium* is a beautiful tool for creating Leaflet maps. I used it to create two separate maps. These mapping visualizations demand a lot from our computers, so I had to minimize our data to 50% to get it to run for our first map. The map is a heat map that shows us that accidents closer to exits cause a higher degree of traffic. Which makes sense as most on and off-ramps are single lane, so it is more likely that an accident will block a more substantial part of the lane hindering traffic to a higher degree. 

![](https://i.imgur.com/z63O8qT.png)

The second map was far more demanding from a computational sense, so I limited our data to 1/8th of its size. Each marker represents a different severity level. As for the markers, purple is the most severe then red-orange and yellow as the lowest severity. If you would like to explore these maps yourself, head over to my notebook for the interactive versions. 

![](https://i.imgur.com/SdIQVDq.png)


The data also included a description column rich with information. These strings were used in the creation of the following masked word clouds. See if you can notice any trends within the language for each severity level. 
![](https://i.imgur.com/aoyosiS.png)
![](https://i.imgur.com/ZUNh1do.png)
![](https://i.imgur.com/DtsYAly.png)
I see our earlier observation of the clustering around excites supported in the word clouds. As the severity increases, I also noticed a shift in language from lane blocked, to lane closed, to the road closed for the most severe. 

Now that our data is ready and we have learned a bit, its time to employ a Random Forest (RF) model to understand the influence each category has on its severity. I chose a RF because the model is resistant to overfitting and does exceedingly well with high dimensionality data as we have here. 

We must first identify our final features and then convert them into dummy variables to assist our random forest. [Dummy variables](https://en.wikipedia.org/wiki/Dummy_variable_(statistics)) are a statistical tool that allows us to break columns like weather into a series of different columns for each category that includes either a 1 or 0. The code I used was as follows 

#What we are trying to predict

target='Severity'

#What we are using to predict our target

features=['Severity', 'Start_Lat', 'Start_Lng', 'Distance(mi)',
          'Side', 'City','Temperature(F)','Wind_Chill(F)', 'Humidity(%)',
          'Pressure(in)', 'Visibility(mi)','Wind_Direction', 'Wind_Speed(mph)',
          'Precipitation(in)','Weather_Condition', 'Junction', 'Crossing', 
          'Traffic_Signal','Sunrise_Sunset','TimeofDay', 'Season', 'Day_Type'] 

#One-Hot Encoding 

df_FF_Dummy=pd.get_dummies(df_FF[features],drop_first=True)

print(df_FF_Dummy.info())

df_FF_ML = df_FF_Dummy.reset_index()

df_FF_ML=df_FF_ML.drop('index',axis=1)

df_FF_ML.fillna(0)

#Train Test Split is a great function to break our data down. i made test 30% of total

y=df_FF_ML[target]

X=df_FF_ML.drop(target, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

Now we must create the model and see the results. 

#Running Model object

clf=RandomForestClassifier(n_estimators=250)

#Train Model with data

clf.fit(X_train,y_train)

#Run Model to predict accident Severity

predictions=clf.predict(X_test)

![](https://i.imgur.com/mHiz0Nq.png)


Now let's see what our model decided was important!

![](https://i.imgur.com/E00fA93.png) 

We see that it truly is location location location. It seems as that this backs up our exit theory from before. Interesting that precipitation scored lower than one might of thought. 

## Results 
I'm quite pleased with these results, given how unbalanced our target variable was. I evaluated the model on several parameters; the model's overall accuracy was just above 76%. Our confusion matrix shows us the number of each prediction and its correctness. The columns counted from severity 1 on the left to severity 4 on the right and the rows being severity 1 on the top and severity 4 on the bottom. The row represents what the model predicted, and the column is what was the actual value. Take the second row; for example, these are severity 2 values. Our model predicted on of these as severity 1 (incorrect), 765 as severity 2 (correct), 173 as severity 3 (incorrect, and 25 as severity 4 (incorrect). 

![](https://i.imgur.com/GXyeHEy.png)

I also had a classifications report made, which shows us the [Precision](https://en.wikipedia.org/wiki/Precision_and_recall), which is the fraction of relevant instances among the retrieved instances, while recall is the fraction of the total amount of relevant instances that were actually retrieved. We also see the [F1 Score](https://en.wikipedia.org/wiki/F1_score), a measure of a test's accuracy. 

![](https://i.imgur.com/qPbLgpK.png)

## Discussion
When it comes to limiting traffic from accidents, decision-makers should focus all their efforts on bottlenecks within our systems. The most noticeable of the jams are our highway exits. This analysis was meant to raise attention to these choke points but not solve them as that would take a more in-depth look at each exit by someone with a higher degree of domain knowledge. The region's decision-makers should employ such a person, as a civil engineer, to evaluate and suggest improvements for the worst of the areas. As far as educating the public, we see that people should keep a closer eye on pressure; when the pressure drops, you can expect the weather to worsen, and humidity levels are a high indicator of traffic than precipitation.

## Conclusion 
There is still much to be done to understand and limit traffic and accidents generally on a larger scale. If you would like to explore this dataset or topic in more depth, several other notebooks reside on [Kaggle.com](https://www.kaggle.com/sobhanmoosavi/us-accidents) for your exploration. Thanks for taking the time to read this and have a good day and drive safely. 
