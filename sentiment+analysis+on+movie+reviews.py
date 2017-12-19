
# coding: utf-8

# In[4]:


import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[ ]:



def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict




# In[ ]:


neg_reviews = []

for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_reviews.append((create_word_features(words), "negative"))
    

print(len(neg_reviews))
    


# In[ ]:


pos_reviews = []

for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append((create_word_features(words), "positive"))
    

print(len(pos_reviews))


# In[ ]:


train_set = neg_reviews[:750] + pos_reviews[:750]
test_set = neg_reviews[750:] + pos_reviews[750:]
print(len(train_set), len(test_set))


# In[ ]:


classifier = NaiveBayesClassifier.train(train_set) #classifier


# In[ ]:


accuracy = nltk.classify.util.accuracy(clf,  test_set)


# In[ ]:


print(accuracy*100)


# In[ ]:


review_dangal = '''Aamir Khan shines in the role of Mahavir Singh Phogat. He is so outstanding in his performance that this could be considered as one of his best works ever.
The opening scene of the film is just a glimpse of Phogat's passion for the sport  wrestling. The first half of Dangal works
exactly like a skilled wrestler  invariably sure of its moves. The film's second half is all about victory. Mahavir's
transformation from the once upset father of a baby girl to the most triumphant one to Geeta's string of losses before
she finally wins a gold during Common Wealth Games to desi akhara being replaced by wrestling mat  every moment works in
contrast to what you had watched earlier. But not even once does the film loses its focus.'''
print(review_dangal)


# In[ ]:


words = word_tokenize(review_dangal)
words = create_word_features(words)


# In[ ]:


classifier.classify(words)


# In[ ]:


transformers5 = '''
The worst movie of this year so far. No logic, no plot, no story, but with Anthony Hopkins. Why did Anthony participated in this dirt ball mind crap, I don't know but it is not enough. This movie is empty, it is a 2 hours long advertisement of various car manufacturers and American military. Worst Transformers ever, please let this garbage just die and don't bother us with actors who just can not save this movie. It was just awful to watch, I couldn't believe the low quality dialogues and just the waste of money for junk CGI no one really cares about. But why did Mr. Hopkins lower himself to participate in this charade of a movie is beyond me.
'''


# In[ ]:


words2 = word_tokenize(transformers5)
words2 = create_word_features(words2)
classifier.classify(words2)

