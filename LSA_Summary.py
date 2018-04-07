#LSA Summarization
import codecs
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

csvfile = codecs.open("C:\\Users\\Amir Ghaderi\\Desktop\\article.txt", mode='r', encoding="utf-8")

text = ""
for i in csvfile:
    text =text + i


#Sentence Tokenizer
sent_tokenize_list = sent_tokenize(text)

#Stopwords
stopset = set(stopwords.words("english"))

#TF-IDF
vectorizer = TfidfVectorizer(stop_words=stopset, use_idf = True, ngram_range=(1,3))
X = vectorizer.fit_transform(sent_tokenize_list)
#X.shape

#SVD
#n_components=X.shape[0]
lsa = TruncatedSVD(n_iter=100)
lsa.fit(X)

#pick the most importance Concept and select the best sentence from that concept
#Then pick the second most important concept and select the second best sentence
#And so on!
