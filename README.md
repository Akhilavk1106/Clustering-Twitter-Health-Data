# Clustering-Twitter-Health-Data
•	Steps 
1.	Data Processing  
Extracted the content of each twitter account, and remove the URL, punctuations and numbers. 
2.	Calculation 
Counted the number of the words and the frequency of each word. Pick the Top 10 words and check if they related to health. 
3.	Elaborated Processing 
Using the nltk package, Removes the stop words. 
Tokenized the text
4.	Visualization 
Using matplotlib, plotted graph of the probability of occurrence for the 10 most common words. 



Task 2: Clustering Task 
•	Steps
1.	Data processing
Firstly, we merge the files together by using package ‘sys’, which will be much easier for us to do the following data processing and identify the labels from different files.We really recommend this way which will save you lots of time.
Secondly, Based on the file we got from task 1, we also remove some special words used frequently in Twitter, like ‘rt’, ‘video’.
 
•	Vectorization of texts – TfidfVectorizer
Made a vector representation of all transformed tweets in the data set, with tf-idf technique. Which will also help us remove some meaningless words.
Remember to limit the features. 
For a huge dataset, chose a proper amount od features will accelerate the time of running.
2.	Principal Component Analysis & K means
Used tf-idf matrix as input to K means. The number of clusters were fixed to 16 as mentioned in the assignment document.
Used PCA as a framework for data dimension reduction and discrete cluster membership indicators for K-means clustering
As mentioned in the assignment document, we used PCA to fix 2 dimensions.
3.	Visualization
Defined color map.
Using matplotlib, plotted the clusters in 2 dimensions.





•	Analysis
1.	Does each Twitter account form its own cluster? 
In order to answering this question, we define the tweets from a same file as a same label, it will be their ‘original labels’. After that, we compare these original labels with clustering labels. 

2.	Why or why not is this the case? 
From the results, we could know that lots of  Twitter accounts didn’t form its own cluster.
The reasons are as following:
a)	The clusters were formed due to familiarity in the text in tweets and since all of them had health related tweets, Its possible that most of content of the Tweets were similar.
b)	All these words were really related to health.
Note: There are still some specific words are only used on social media. We need to find a way to identify most of them and remove this noise to enhance the accuracy of clustering results.

