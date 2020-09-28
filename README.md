# Fairfeild Conneticut Traffic Analysis: Folium, RFC & WordCloud
## THE PROBLEM 
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
All of my Notebook work was done on the Kaggle platform then downloaded and pushed to this Github repository. The data was derived form a Kaggle dataset by the name of [*US Accidents (3.5 million records)*](https://www.kaggle.com/sobhanmoosavi/us-accidents) as well as a small set of png files for wordcloud masking. 

while the data set encompassed the 48 contiguous Sates that make up the main body of the US the scope of this analysis was focused on Conneticuts Fairfeild county. Fairfeild is located in the Southwestern corner of the State and is the most populous and fast growing county which is why it was chosen for this anlysis. ![](https://en.wikipedia.org/wiki/Fairfield_County,_Connecticut#/media/File:Map_of_Connecticut_highlighting_Fairfield_County.svg)




Methodology section which represents the main component of the report where you discuss and describe any exploratory data analysis that you did, any inferential statistical testing that you performed, if any, and what machine learnings were used and why.
Results section where you discuss the results.
Discussion section where you discuss any observations you noted and any recommendations you can make based on the results.
Conclusion section where you conclude the report.
