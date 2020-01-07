#LSA Summarization
#Amir Ghaderi

#Import Libraries

import codecs
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.extmath import randomized_svd

#Read in article
csvfile = codecs.open("file location",'r','utf-8')
Article = ""
for line in csvfile:
	Article = Article + line

#split text into sentences
sent_tokenize_list = sent_tokenize(Article)\

#Stopwords
stopset = set(stopwords.words("english"))

#TF-IDF
vectorizer = TfidfVectorizer(stop_words=stopset,use_idf=True,ngram_range(1,3))
X = vectorizer.fit_transform(sent_tokenize_list)
X_T = np.transpose(X)

#SVD/LSA
U, Sigma, VT = randomized_svd(X_T, n_components=100, n_iter=100, random_state = None)

#Sentence Selection

k = 6
temp_k = k
i = 0
output = []
index_list =[]

if k>=3:
	while k != 0:
		if i == 0:
			dic = {}
			for j in range(0,len(VT[i])):
				dic[j] = VT[i][j]
			dic_sort = sorted(dic.items(), key=operator.itemgetter(1))
			index1 = dic_sort[-1][0]
			index2 = dic_sort[-2][0]
			index3 = dic_sort[-3][0]
			list = [index1,index2,index3]
			list = sorted(list)
			for t in list:
				output.append(sent_tokenize_list[t])
				index_list.append(t)
			k = k - 3
		if k<=0:
			break
		elif i == 1 and temp_k != 4:
			dic = {}
			for j in range(0,len(VT[i])):
				dic[j] = VT[i][j]
			dic_sort = sorted(dic.items(), key=operator.itemgetter(1))
			temp_list = []
			count = 0
			for p in range(-1,-len(VT[i]),-1,-1):
				if count < 2:
					if dic_sort[p][0] not in index_list:
						temp_list.append(dic_sort[p][0])
						index_list.append(dic_sort[p][0])
			list = sorted(temp_list)
			for t in list:
				output.append(sent_tokenize_list[t])
			k=k-2
		if k <=0:
			break
		elif i >=2:
			max = -9999999999
			for j in range(0,len(VT[i])):
				if VT[i][j] > max and j not in index_list:
				index = j
				max = VT[i][j]
				index_list.append(j)
			output.append(sent_tokenize_list[index])
			k=k-1
		i = i + 1

if temp_k<3:
	dic = {}
	for j in range(0,len(VT[0])):
		dic[j] = VT[0][j]
	dic_sort = sorted(dic.items(), key=operator.itemgetter(1))

	index1 = dic_sort[-1][0]
	index2 = dic_sort[-2][0]
	list = [index1,index2]
	list = sorted(list)
	if k ==1:
		output.append(sent_tokenize_list[list[0]])
	if k ==2:
		output.append(sent_tokenize_list[list[0]])
		output.append(sent_tokenize_list[list[1]])
		
