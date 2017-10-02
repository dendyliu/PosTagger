import nltk
import pprint
import nltk.corpus, nltk.tag, itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

#Function buat extract features	
def features(sentence, index):
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }

#Function buat lepas ambil word dari tagged_sentences
def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]

#Function buat ubah tagged_sentences jadi dataset dimana X  = hasl extract fitur , Y = labelnya atau tagnya
def transform_to_dataset(tagged_sentences):
    X, y = [], []
 
    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])
 
    return X, y

def run_pos_tag(corpus_sentences):
	#Data isinya sentence yang dah ada tag nya
	tagged_sentences = corpus_sentences
	# Split the dataset for training and testing
	cutoff = int(.75 * len(tagged_sentences))
	training_sentences = tagged_sentences[:cutoff]
	test_sentences = tagged_sentences[cutoff:]
	#print bnyk data training ama data test
	print len(training_sentences)  
	print len(test_sentences)      
	#Extract fiture manual dari  training sentence ama test sentence
	X, y = transform_to_dataset(training_sentences)
	X_test, y_test = transform_to_dataset(test_sentences)
	#TAGGER CLASSIFIER=================================================
	#hmm_trainer =  nltk.tag.hmm.HiddenMarkovModelTrainer()
	#hmm_tagger = hmm_trainer.train_supervised(brown_train)
	#hmm_tagger.test(brown_test)
	unigram_tagger = nltk.tag.UnigramTagger(training_sentences)
	print "unigram tagger accuracy : ",unigram_tagger.evaluate(test_sentences)
	#Classifier biasa=======================================================
	clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', DecisionTreeClassifier(criterion='entropy'))
    #('classifier', SVC(kernel='linear'))
    #('classifier', RandomForestClassifier())
    #('classifier', MLPClassifier())
	])
	clf.fit(X[:8000], y[:8000])
	print 'Training Classifier completed'
	print "Accuracy:", clf.score(X_test[:2000], y_test[:2000])
#ISI 3 CORPUS DATA brown,conll2000,treebank
brown_review_sents = nltk.corpus.brown.tagged_sents(categories=['reviews'])
brown_lore_sents = nltk.corpus.brown.tagged_sents(categories=['lore'])
brown_news_sents = nltk.corpus.brown.tagged_sents(categories=['news'])

brown_sents = list(itertools.chain(brown_review_sents[:2000], brown_lore_sents[:2000], brown_news_sents[:2000]))
 
conll_sents = nltk.corpus.conll2000.tagged_sents()[:6000]
treebank_sents = nltk.corpus.treebank.tagged_sents()[:6000]
run_pos_tag(brown_sents)