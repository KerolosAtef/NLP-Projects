# **Text Classification Project**
**!! NOTE !!**
- **If it is required to run the code from colab, change run time type to GPU.**
-  [Text_Classification_Assignment](https://colab.research.google.com/drive/1-pq3XEWFu0r_9veYj5pJy9l1wfjlbJdH?usp=sharing#scrollTo=xENuM02UbaYG "Text_Classification_Assignment")

## **Table of Contents**

- [Overview](#Overview)
- [Methodologies](#Methodologies)
	- [Data Preparing and Preprocessing](#Data-Preparing-and-Preprocessing)
	- [Transformers](#Transformers)
		- [BOW](#BOW)
		- [TF-IDF](#TF-IDF)
			- [N-Gram](#N-Gram)
		- [LDA](#LDA)
		- [Doc2Vec](#Doc2Vec)
		- [Word2Vec](#Word2Vec)
		- [Glove](#Glove)
		- [Bert](#Bert)
	- [Training](#Training)
	- [Validation](#Validation)
	- [Testing](#Testing)
	- [Error analysis](#Error-analysis)
		- [Reduce Number of words](#Reduce-Number-of-words)
		- [Weights of words](#Weights-of-words)
		- [Grid Search](#Grid-Search)
- [Conclusion](#Conclusion)
- [Dependencies](#Dependencies)



# Overview
--------------------------------
In this project, it is required to choose 5 random books from the Gutenburg library which are of the same genre, semantically the same, and are by 5 different authors. 

After that, it's required to build a data frame that has 200 random partitions of each book, and each partition should have 100 words.

Then we should apply some preprocessing to clean the books, besides splitting the data into training and testing datasets.
Furthermore, it is required to apply some transformations such as BOW, TFIDF, N-Grams, or Bert, in addition, build some classification models such as SVM, KNN, Naive Bayes, or Decision Trees to classify the book name.

In addition to that, it is required to evaluate all models and choose the champion model among them that achieves the highest score, then apply some error analysis to this model.

# Methodologies
--------------------------------
## Data Preparing and Preprocessing
At this step, our task is to have a look at the data, explore its main characteristics: size & structure (how sentences, paragraphs, text are built), finally, understand how much of this data is useful for our needs? We started by reading the data.
•	We used nltk library to access Gutenberg books. we chose the IDs of five different books but in the same genre.

```python
nltk.download('gutenberg')

books_idx=[12,8,11,0,5]
selected_books=[]
for idx in books_idx :
  selected_books.append(books_names[idx])
print(selected_books)

book_contents=[]
for book_name in selected_books:
  book_contents.append(nltk.corpus.gutenberg.raw(book_name))
book_contents
```
•	We displayed the data to see if it needs cleaning. We found the output of the data like this:

![image](https://drive.google.com/uc?export=view&id=19eo5melDSUVM2WwKMtrSIgfZZxo-Uckq)


•	Then, we cleaned the data from any unwanted characters, white spaces and stop words. 
•	We tokenized the data to convert it into words
•	We converted the cleaned data in lower case.
•	Then, we lemmatized words and switched all the words to its base root mode .
	**There is a function called (clean_text) that takes the uncleaned books as an input and returns the book cleaned.**

```python
cleaned_books_contents=[]
for book in book_contents :
  cleaned_books_contents.append(clean_text(book))
```
•	We labeled the cleaned data of each book with the same name.
•	Then, we chunked the cleaned data for each book into 200 partitions, each partition contains 100 words. So, now we have (1000x2) Data frame.

**There is a function (book_samples) that takes the cleaned book as an input and returns 200 partitions for book , and each one has 100 words.**

```python
samples_of_books=[]
for cleaned_book in cleaned_books_contents :
  samples_of_books.append(book_samples(cleaned_book))
samples_of_books
```
```python
# Creating a data frame with the required partitions
data_frame =pd.DataFrame()
data_frame['Sample of the book']=[item for sublist in samples_of_books for item in sublist]
target=[[books_names[i]]*min(200,len(samples_of_books[i])) for i in range(len(selected_books)) ]
data_frame['Book_name']=[item[:-4] for sublist in target for item in sublist]
data_frame
```
![image](https://drive.google.com/uc?export=view&id=188DNKMqQyTgcVOsbQ1eLou5so_0Hx3KO)

## Transformers
- It is one of the trivial steps to be followed for a better understanding of the context of what we are dealing with. After the initial text is cleaned and normalized, we need to transform it into their features to be used for modeling.

- We used some particular methods to assign weights to particular words, sentences or documents within our data before modeling them. We go for numerical representation for individual words as it is easy for the computer to process numbers.

- Before starting to transform words. We split the data into training and testing, to prevent data leakage.

### BOW
•	A bag of words is a representation of text that describes the occurrence of words within a document. We just keep track of word counts and disregard the grammatical details and the word order.

•	As we said that we split the data. So, we applied BOW on training and testing data. 

•	We transformed each sentence as array of word occurrence in this sentence.

```python
from sklearn.feature_extraction.text import CountVectorizer

BOW = CountVectorizer()
BOW_train = BOW.fit_transform(X_train)
BOW_test = BOW.transform(X_test)
```

### TF-IDF
TF-IDF (term frequency-inverse document frequency) is a statistical measure that evaluates how relevant a word is to a document in a collection of documents. This is done by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents.

 #### N-Gram
We applied in the uni-gram and bi-gram on training and testing sets by TFIDF vectorizer. Which creates a dictionary containing n-grams as keys to the dict and a list of words that occurs after the n-gram as values.
```python
def tfidf_ngram(n_gram,X_train=X_train,X_test=X_test):
    vectorizer = TfidfVectorizer(ngram_range=(1,n_gram))
    x_train_vec = vectorizer.fit_transform(X_train)
    x_test_vec = vectorizer.transform(X_test)
    return x_train_vec,x_test_vec 

 # Applying unigram and bigram for transformation
X_trained1g_cv,X_test1g_cv = tfidf_ngram(1,X_train=X_train,X_test=X_test)
X_trained2g_cv,X_test2g_cv = tfidf_ngram(2,X_train=X_train,X_test=X_test)
```

### LDA
•	LDA is used to classify text in a document to a particular topic. It builds a topic per document model and words per topic model. It is also used as dimensionality reduction.

•	We used LDA as a transformer after vectorization used in BOW, because in LDA it can not vectorize words.
```python
# Using the BOW vectorizer from the function of the BOW above.
lda = LDA()
lda_train = lda.fit_transform(BOW_train.toarray(), y_train)
lda_test = lda.transform(BOW_test.toarray())
```
### Doc2Vec
•	Doc2Vec is a method for representing a document as a vector and its build on word2vec approach.

•	We trained a model from scratch to embed each sentence or paragraph of the data frame as a vector of 50 elements.

### Word2Vec
•	Word2vec is a method to represent each word as a vector.

•	We used a pretrained model “Google-news-vector-negative300”.

**It's required to download the pretrained model from this command**
```python
!gdown --id 0B7XkCwpI5KDYNlNUTTlSS21pQmM
```


### Glove
•	Global vector for word representation is un supervised learning algorithm for word embedding.

•	We trained a GloVe model on books  data, that represent each word in a 100x1 Vector. We took the data frame after cleaning and get each paragraph and passed it to the corpus. After that we trained the model on each word.

•	We used also a pretrained model “glove.6B.100d.txt. a file that containing 4000000 words, each word represented by a 100x1 vector. Then, on each word of sentence in the data frame, we replaced it with its vector representation.

**It is required to install glove_python then glove-python-binary. **

**It is required to download the pretrained model from the link below.**

```python
!gdown --id 1tfswA5-s4LkMTLWxX9PaaX7z6zbJD54a
```
```python
 #https://drive.google.com/file/d/1tfswA5-s4LkMTLWxX9PaaX7z6zbJD54a/view?usp=sharing
embeddings_index = dict()
f = open('/content/glove.6B.100d.txt', encoding="utf8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
```
### Bert
•	BERT (Bidirectional Encoder Representations from Transformers) is a highly complex and advanced language model that helps people automate language understanding.

•	BERT is the encoder of transformers, and it consists of 12 layers in the base model, and 24 layers for the large model. So, we can take the output of these layer as embedding vector from the pretrained model. 

•	There are three approaches in the embedding vectors: concatenate the last four layers, sum of last four layers, or embed the full sentence by taking the mean of the embedding vectors of the tokenized words

•	As the first two methods require computational power, we used the third one which takes the mean of columns of each word and each word is represented as 768x1 vector. so, the whole sentence at the end is represented as 768x1 vector.

**For installtion, it is required to install transformers then simpletransformers .**

**Also It is needed to install pytorch with cuda and tensorflow with cuda.**


**NOTE !**
- If it is needed to run it from colab, change run time type to GPU.
- If it is needed to run the CPU instead of GPU, you have to change it from the function which called (sentence_embedding_BERT).

```python
  # change device = 'cuda' to device='CPU' 
    tokens_tensor = torch.tensor([indexed_tokens],device='cuda')
     segments_tensors = torch.tensor([segments_ids],device='cuda')
  # remove this line
     model.cuda()
```

## Training
--------------------------------
After splitting and transforming the data, and extracting features from it we applied many learning algorithms to the training data for classification such as:

•	SVM is a supervised machine learning algorithm that separates classes using hyperplanes.

•	Gaussian NB is special type of Naïve Bayes algorithm that perform well on continuous data.

•	KNN is a non-parametric supervised algorithm. Despite its simplicity it can be highly competitive in NLP applications.

•	Decision Tree uses a tree-like model in to take a decision and studying its consequences.

So, we have 28 models on all of our transformation methods.


## Validation
--------------------------------
We applied 10-folds cross validation to estimate the skills of our machine learning models on different combinations of validation and training datasets.

![image](https://drive.google.com/uc?export=view&id=11blcfTPxrRvdQdeHwKLNQPmRC4lqLZPz)


## Testing
--------------------------------
After training and validation of our models we tested all our models on testing data and we found the results are like: 

![image](https://drive.google.com/uc?export=view&id=1NpSZjAeM9xAQ_rZwSeP4UdChJTaTYenG)

- **BERT as a stand-alone classifier**
    - BERT can be used as the main classifier by finetuning the model on our data set.
    - We used this classifier from Hugging face library which called simple transformer.
    - Training Accuracy is 0.93875
    - Testing Accuracy is 0.89

**From these results, we concluded that the champion model is SVC, and champion embedding is TF-IDF uni-gram. Achieving training accuracy of 100% and testing accuracy of 99.5%. We applied 10 folds cross validation on champion model:**

![image](https://drive.google.com/uc?export=view&id=1iplJXuV1jPNiIznh5coaaG_NAWcNY_uk)

## Error analysis
--------------
### Reduce Number of words
We reduced the number of words in each sentence to test if the accuracy of the champion model will decrease, increase or will still the same.
```python
[0.815, 0.735, 0.665, 0.645, 0.615]
```
We noticed that the accuracy decreased by decreasing number of words in each partition and this make sense because the model can’t classify which class when number of words(features) is small.

### Weights of words
At first, we explored the weights of word examples in correct and wrong book classification to make sure that everything is working fine.

### Grid Search
- We used grid search to tune hyperparameters and calculate the optimum values of hyperparameters of the champion model.

- We found that the best hyperparameters value for kernel is linear and for regularization coefficient is 1 .

## Conclusion  
---------------------------------
To wrap up, we made 32 model on different transformation methods and it was obvious that SVM perform better than other models, this is because the data is slightly few, and SVM performs better when data is small. When comparing transformation methods, it clear that TF-IDF uni-gram is trained better in most of the models, because as the length of n-grams increase, the frequency of finding this n-grams again decreases.

# Dependencies
----------------------------------------
- nltk
- glove_python
-  glove-python-binary
- transformers
- simpletransformers
- numpy
- pandas
- seaborn
- matplotlib
- cuda == 11.2 for tensorflow
- cuda == 11.3 for pytorch
- torch ==1.11
- tensorflow == 2.9
- sklearn
- gensim


```python

```
