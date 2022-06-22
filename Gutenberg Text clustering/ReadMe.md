# Table of Contents
[1. Introduction](#_Toc106538166)

[2. Methodology](#_Toc106538167)

[2.1 data preparing and Preprocessing](#_Toc106538168)

[2.2 Transformation methods](#_Toc106538169)

[2.2.1 BOW](#_Toc106538170)

[2.2.2 TF-IDF](#_Toc106538171)

[2.2.2.1 n-gram](#_Toc106538172)

[2.2.3 Doc2Vec](#_Toc106538173)

[2.2.4 Word2Vec](#_Toc106538174)

[2.2.5 GloVe](#_Toc106538175)

[2.2.6 BERT](#_Toc106538176)

[2.2.7 LDA (Latent Dirichlet Allocation)](#_Toc106538177)

[2.2.8 FastText](#_Toc106538178)

[2.3 Training](#_Toc106538179)

[2.4 Evaluation](#_Toc106538180)

[2.4.1 K-means](#_Toc106538181)

[2.4.2 EM](#_Toc106538182)

[2.4.3 Hierarchical Clustering](#_Toc106538183)

[2.4.4 Choosing Champion Model.](#_Toc106538184)

[2.5 Error analysis](#_Toc106538185)

[2.5.1 showing the most frequent words](#_Toc106538186)

[2.5.2 Cosine Similarity](#_Toc106538187)

[3. Conclusion](#_Toc106538188)







# 1. Introduction
Following the last assignment, we took the same cleaning and preprocessing code and we adjusted it to follow this assignment’s rules. We chose five books of different genres. we put into consideration that these books are large to generate enough data to make a clustering model.

# 2. Methodology
## `      `2.1 data preparing and Preprocessing
`		`At this step, our task is to have a look at the data, explore its main characteristics: size & structure (how sentences, paragraphs, and text are built), and finally, understand how much of this data is useful for our needs? We started by reading the data.

- We used the nltk library to access Gutenberg books. we chose the IDs of five different books but in the same genre.
- We displayed the data to see if it needs cleaning. We found the output of the data like this:

![](https://drive.google.com/uc?export=view&id=13SdS7xv_83P4AXXNHyMxFCNrAaqcSIaY)


- Then, we cleaned the data from any unwanted characters, white spaces, and stop words. 
- We tokenized the data to convert it into words
- We converted the cleaned data in lower case.
- Then, we lemmatized words and switched all the words to their base root mode 
- We labeled the cleaned data of each book with the same name. 
- Then, we chunked the cleaned data for each book into 200 partitions, each partition containing 100 words. So, now we have a (1000x2) Data frame.

`                      `!![](https://drive.google.com/uc?export=view&id=1SkWL9oXZIMybCHPu7yfAPEVJw5HXZdR3)

`                                                                 `Fig.1 cleaned data

## `     `2.2 Transformation methods
It is one of the trivial steps to be followed for a better understanding of the context of what we are dealing with. After the initial text is cleaned and normalized, we need to transform it into its features to be used for modeling.

We used some methods to assign weights to particular words, sentences, or documents within our data before modeling them. We go for numerical representation for individual words as it’s easy for the computer to process numbers.

Before starting to transform words. We split the data into training and testing, to prevent data leakage.

### `	`2.2.1 BOW
- A bag of words is a representation of text that describes the occurrence of words within a document. We just keep track of word counts and disregard the grammatical details and the word order
- As we said that we split the data. So, we applied BOW to training and testing data. 
- We transformed each sentence as an array of wooccurrencesnce in this sentence.

![](https://drive.google.com/uc?export=view&id=1EYO3Wurl9Nx32gcg8NnhundKAkXqdH12)
### `	`2.2.2 TF-IDF
- TF-IDF (term frequency-inverse document frequency) is a statistical measure that evaluates how relevant a word is to a document in a collection of documents. This is done by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents.
### `		`2.2.2.1 n-gram
- We applied in the unigram on training and testing sets. Which creates a dictionary containing n-grams as keys to the diet and a list of words that occur after the n-gram as values.

`            `![](https://drive.google.com/uc?export=view&id=11gktQM9MIGeqHY4pdxanEm7XMsjmkH8s)

### `             `2.2.3 Doc2Vec
- Doc2Vec is a method for representing a document as a vector and is built on the word2vec approach.
- We trained a model from scratch to embed each sentence or paragraph of the data frame as a vector of 50 elements.

![](https://drive.google.com/uc?export=view&id=18lthhjxyskVPKUOzEXOzwn_3eRuQx3j8)

### `             `2.2.4 Word2Vec
- Word2vec is a method to represent each word as a vector.
- We used a pre-trained model “word2vec-google-news-300”.

![](https://drive.google.com/uc?export=view&id=1NM8COYTYBDAo7d90NWhoB1FYQuDhMK_1)
### `	`2.2.5 GloVe
- Global vector for word representation is an unsupervised learning algorithm for word embedding.
- We trained a GloVe model on books’ data, that represents each word in a 300x1 Vector. We took the data frame after cleaning and get each paragraph and passed it to the corpus. After that,t we trained the model on each word.
- We used also a pre-trained model “glove-wiki-gigaword-300”. Each word is represented by a 300x1 vector. Then, on each word of a sentence in the data frame, we replaced it with its vector representation.

![](https://drive.google.com/uc?export=view&id=1L6MSQyWAOpjCmeqydbcLDbS8B-szKeXV)

### `             `2.2.6 BERT
- BERT (Bidirectional Encoder Representations from Transformers) is a highly complex and advanced language model that helps people automate language understanding.
- BERT is the encoder of transformers, and it consists of 12 layers in the base model, and 24 layers for the large model. So, we can take the output of these layers as an embedding vector from the pretrained model. 
- There are three approaches to the embedding vectors: concatenate the last four layers, the sum of the last four layers, or embed the full sentence by taking the mean of the embedding vectors of the tokenized words
- As the first two methods require computational power, we used the third one which takes the mean of columns of each word and each word is represented as a  768x1 vector. so, the whole sentence at the end is represented an as a 768x1 vector 

![](https://drive.google.com/uc?export=view&id=1VEXnH8jHfOZCo0-OeIk_0vBBK9fFqNAe)

### 2.2.7 LDA (Latent Dirichlet Allocation)
- It is a [generative statistical model](https://en.wikipedia.org/wiki/Generative_model "Generative model") that explains a set of observations through [unobserved](https://en.wikipedia.org/wiki/Latent_variable "Latent variable") groups, and each group explains why some parts of the data are similar. LDA is an example of a [topic model](https://en.wikipedia.org/wiki/Topic_model "Topic model").
- We used LDA as a transformer after vectorization used in BOW because LDA can’t vectorize words. So, we needed to use it after BOW.

![](https://drive.google.com/uc?export=view&id=1dULAx72jxnEc2ePaf8Of1qk1PHIYk9CA)

- We visualized the results of topic modeling Using LDA, it appears as follows

![](https://drive.google.com/uc?export=view&id=1svGkMs-8D0ZAm8DVhnjFpoOrEtnDiEB_)



![](https://drive.google.com/uc?export=view&id=1hDcAgLDxiiKqDnMdpkFYPq7Q0aCb0Zu7)

![](https://drive.google.com/uc?export=view&id=1oi2LPbtI5Y18pD9OYS7TbYTCtd7gvEZW)

![](https://drive.google.com/uc?export=view&id=1ThmOllLJLSjKmxVoAJ6LagWnM6G8tjZK)

![](https://drive.google.com/uc?export=view&id=1EbSfvgX626NuwQZ8hKNBoGY0L6HIKZfo)

- We also measured the coherence per topic of the LDA model:

`                `![](https://drive.google.com/uc?export=view&id=1IC9IVsWJBwY1NNlqKN3W8MH-8f0u4T27)

And the model coherence:

`                        `![](https://drive.google.com/uc?export=view&id=1nix8Vw1ZAjKiNIwErduBjb_44JNB7Xl5)




### 2.2.8 FastText
- FastText is a library for learning word embeddings and text classification. The model allows one to create unsupervised learning or supervised learning algorithms for obtaining vector representations for words.
- We loaded a pre-trained model from genism API ‘fasttext-wiki-news-subwords-300’.

![](https://drive.google.com/uc?export=view&id=1qnJv0pkHFXvRi-uhpTZ1gKCrZBfWzC8W)


## `      `2.3 Training
After splitting and transforming the data, and extracting features from it we applied many learning algorithms to the data for Clustering such as:

- Kmeans is an unsupervised machine learning algorithm in which each observation belongs to the cluster with the nearest mean.
- EM clustering is to estimate the means and standard deviations for each cluster to maximize the likelihood of the observed data.
- Hierarchical clustering is an algorithm that groups similar objects into groups. The endpoint is a set of clusters, where each cluster is distinct from the other cluster, and the objects within each cluster are broadly like each other.

` `So, we have 24 models for all our transformation methods.     

##
## 2.4 Evaluation
After the training phase, we used many metrics to measure the performance  of the clustering algorithm with each transformation method: 
### 2.4.1 K-means
`       `K-means is an unsupervised machine learning algorithm in which each observation belongs to the cluster with the nearest mean.

#### `      `2.4.1.1 Elbow Method
`                  `![](https://drive.google.com/uc?export=view&id=18_1vfN1X01L8ktHZhWKLX8hFa_k9ihIw)

**As shown in the figure that the best transformation methods with k=5 are Doc2vec, TF-IDF, BERT, and LDA.** 
#### 2.4.1.2 silhouette Score  

#### ![](https://drive.google.com/uc?export=view&id=1xfNlYq8hTsMVuz5kYWGmZ4RKxA5OBN8E)  


**As shown in the figure that the best transformation methods with silhouette score when K= 5 are Doc2vec, TF-IDF, and LDA.** 
#### Before comparing the predicted clusters with the human labels, we should take care of the mapping between cluster names to get the right Kappa Score.<br>
This is an Example before mapping 
#### ![](https://drive.google.com/uc?export=view&id=1jtQsdHdFiZ_R6_nBVbOLYMd-2GDvjwl-)

![](https://drive.google.com/uc?export=view&id=1xW3P-PyA_SZn-3EPdq6XuFvmPsJjHokk)<br>


![](https://drive.google.com/uc?export=view&id=1QlkjHyAjIGo_Wy-ukKl9oVZ2JveQDgxz)
























#### 2.4.1.3 Kappa Score  

![](https://drive.google.com/uc?export=view&id=13Jthd0lW_cGAhlnjPxQRbBuzFLvKqbZb)

The Highest Kappa score is Doc2Vec with 99.25%.





#### 2.4.1.4 Visualize clusters using Doc2Vec

Clusters with actual Labels:

![](https://drive.google.com/uc?export=view&id=1Zyi9_eGthgTmvxahH-YZP6U0mlogveHi)

Clusters with Kmeans: 

![](https://drive.google.com/uc?export=view&id=1qPz_mFJx2fk2tZSRZWgqSdIIOcLSmTPh)


### 2.4.2 EM
EM clustering is to estimate the means and standard deviations for each cluster to maximize the likelihood of the observed data.

2.4.2.1 Silhouette Score


![](https://drive.google.com/uc?export=view&id=10VdClMiydLr-jxrHz6df5jZiSeRZDCGa)



The Highest silhouette score is LDA when k=4.

2.4.2.2 BIC Score

![](https://drive.google.com/uc?export=view&id=1sadya6rRxwKMB6Hyqs7SJv4CP6d80PEl)
The lowest BIC Score is Fast text with k=2.






2.4.2.3 Kappa Score

![](https://drive.google.com/uc?export=view&id=1OI7ECTL9iOjURb052yoDMGXvDMZdaGA8)

**The highest Kappa score is Doc2vec with 99.6%.**





2.4.2.4 Using PCA with the highest silhouette score to visualize clusters

Doc2Vec (Highest Kappa Score):

![](https://drive.google.com/uc?export=view&id=1DjQyRmH_a28gGP_B5XjtuUmYhFPbAH-Q)

LDA (Highest Silhouette Score): 

![](https://drive.google.com/uc?export=view&id=1U9Ff66ad7vtGDcJ7tEXhn4RZGNMXJbP-)



Fast Text (Lowest BIC Score):

![](https://drive.google.com/uc?export=view&id=1Nk7JKBQPkLY63aVU4jw223IqZfCzc34p)
###





According to human labels, so the best model is Doc2vec with k=5.
### 2.4.3 Hierarchical Clustering
Hierarchical clustering is an algorithm that groups similar objects into groups. The endpoint is a set of clusters, where each cluster is distinct from the other cluster, and the objects within each cluster are broadly like each other.
#### 2.4.3.1 Elbow Method

![](https://drive.google.com/uc?export=view&id=1W58wM8M7fGq3n3Ku7tHfx-yZTVnPH9J4)

The majority of the models voted for k =4. 


#### 2.4.3.2 silhouette score

![](https://drive.google.com/uc?export=view&id=1PeHdRrGC2GpFIckULawsuxhjBmq_t9Dv)

The majority of the models voted for k =5. 




#### 2.4.3.3 Kappa score

![](https://drive.google.com/uc?export=view&id=1JN8Wa-a8LPKUjQJi3XpLX_0PH2Q033DN)

The highest Kappa score is Doc2Vev With 99.25%.

**So, this is the Hierarchy champion model**







**Dendrogram of the result** 


![](https://drive.google.com/uc?export=view&id=1y9_tHkfRjPuLkFb4azUwMjmd5QObJ_-8)










### 2.4.4 Choosing Champion Model.
As Doc2Vec achieved the best scores in all the models. we applied the three models with doc2vec to choose the champion model

`                    `![](https://drive.google.com/uc?export=view&id=1FmLq6uT8zMXuAmA2mbJwzZX-TqHXlkh6)

**As shown in the figure, Doc2Vec with the EM cluster has the best score among all clusters.**












## 2.5 Error analysis
### 2.5.1 showing the most frequent words
- We draw the most frequent words in the wrong sample. the champion model predicted that this sample belong to cluster 3 and the actual label was 2, and it appeared as follows:

`              	  `![](https://drive.google.com/uc?export=view&id=1NaqYJMtWT0Kxf_qwpQRFWCFZROpfyrl8)

- The most frequent words in the actual clusters (2 and 3):

`                  `![](https://drive.google.com/uc?export=view&id=1CVWMlvGJGdSGpaGWP7KDGiIjG1ZSD3WD)

- we can observe that the most frequent word in the wrong predict sample is almost similar to cluster 3. Like the word “ship” for example. This can tell us why it was misclustered.
- We also wanted to make sure by a seen metric. So, we got the 10 most frequent words in this sample, and how frequent these words were in the true book and the predicted book.

![](https://drive.google.com/uc?export=view&id=Awli_HSQpbgVNfaUReTYGQ9D8NTobce_)

- We observed that 5 of the most frequent word in the sample are more frequent in the predicted book. 
- When the diff column is negative, this shows that this word is more frequent in the predicted label than in the actual label.
###
### 2.5.2 Cosine Similarity
- We calculated the mean of the actual and predicted books as a vector of 50x1 and calculated cosine similarity between them and the wrong samples. We put the results into a DataFrame, it appeared as follows: 

![](https://drive.google.com/uc?export=view&id=1Nd-xej2KoBZjvVkQf_WmAPvfUDwmtg7h)

Figure: cosine similarity of true predicted samples

- As shown in the figure, we noticed that the true predicted samples have large cosine similarities.
- Then we print three random wrong samples to see the results.

![](https://drive.google.com/uc?export=view&id=1OgMH-gSz8mtEL2NnoNe0NQh80KpH7KTb)

- This shows that the similarity between book representation and the wrong predicted labels is larger than the actual labels. This gives us an intuition why the machine fails to predict it.

# 3. Conclusion  
- After cleaning and preprocessing, we used 8 different transformation methods to apply text clustering. Then, we applied 3 different clustering algorithms to these 8 methods. This resulted in 24 models to evaluate the best transformation method that can work with clustering in our case. As shown in the report, Doc2Vec performed better with all the 3 algorithms. After comparing these 3 models on Doc2Vec, we found that EM with Doc2Vec is the champion model. After this, we performed error analysis using cosine similarity and the most frequent words in the mislabeled documents. And the result shows that most of the mislabeled documents have words that are more frequent in the prediction, not the actual labels.


