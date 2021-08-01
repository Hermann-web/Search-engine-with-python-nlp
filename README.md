# SearchEngine


##Prerequistes
*   Python >=3.7
*   NLTK
*   Scikit-learn
*   Pandas

##data
Files used in the notebook aare stored in the folder data

# **1: Créer les keywords à partir d'une phrase en se basant sur les mots d'un dictionnaire et un corpus de texte en passant par la tokenization, la correction, la lemmatization et le removeStopWords**
---
##fonction: SENTENCE_TO_CORRECT_WORDS
--- 

```
import re
from collections import Counter
import unicodedata, re, string
import json 


def get_dico():
    textdir = "liste.de.mots.francais.frgut_.txt"
    try:DICO = open(textdir,'r',encoding="utf-8").read()
    except: DICO = open(textdir,'r').read()
    
    textdir = 'corpus_.txt'
    try:CORPUS = open(textdir,'r',encoding="utf-8").read();found=True
    except:pass
    try: CORPUS = open(textdir,'r').read();found=True 
    except: pass
    CORPUS = open(textdir,'r',encoding='cp1252').read();found=True 

    
    
    #WORDS = Counter(words( 'manger bouger difference update All edits that are one edit away from `word`. The subset of `words` that appear in the dictionary of WORDS '))
    
    return DICO+CORPUS


def remove_accents(input_str):
    '''
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii
    '''
    """This method removes all diacritic marks from the given string"""
    norm_txt = unicodedata.normalize('NFD', input_str)
    shaved = ''.join(c for c in norm_txt if not unicodedata.combining(c))
    return unicodedata.normalize('NFC', shaved)

def clean_sentence(texte):
    # Replace diacritics
    texte = remove_accents(texte)
    # Lowercase the document
    texte = texte.lower()
    # Remove Mentions
    texte = re.sub(r'@\w+', '', texte)
    # Remove punctuations
    texte = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', texte)
    # Remove the doubled space
    texte = re.sub(r'\s{2,}', ' ', texte)
    #remove whitespaces at the beginning and the end
    texte = texte.strip()
    
    return texte

'''
#pas cool car il ya des nombres importants
def tokenize_sentence3(texte):
    #retourner les groupes d'alphabets
    return re.findall(r'\w+', texte.lower())
'''
'''
#inutile ici car sa VA est qu'ik decoupe en phrases
def tokenize_sentence2(texte):
        #clean the sentence
    blob_object = TextBlob(texte)
        #tokenize
    liste_words = blob_object.words
        #return 
    return liste_words
'''
def tokenize_sentence(texte):
        #clean the sentence 
    texte = clean_sentence(texte)
        #tokenize 
    liste_words = texte.split()
        #return 
    return liste_words

def strip_apostrophe(liste_words):
    get_radical = lambda word: word.split('\'')[-1]
    return list(map(get_radical,liste_words))

def pre_process(sentence):
    #remove '_' from the sentence 
    sentence = sentence.replace('_','')
    
    #get words fro the sentence 
    liste_words = tokenize_sentence(sentence)
    #cut out 1 or 2 letters ones 
    liste_words = [elt for elt in liste_words if len(elt)>2]
    #prendre le radical après l'apostrophe
    liste_words = strip_apostrophe(liste_words)
    print('\nsentence to words : ',liste_words)
    return liste_words

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)
    
def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])




def DICO_ET_CORRECTEUR():
    "cette fonction retourne la liste des mots de dictionnaire"
    DICO = get_dico()
    WORDS = Counter(pre_process(DICO)) #Counter prends un str et retourne une sorte de liste enrichie
    "correction des mots "
    N = sum(WORDS.values())
    P = lambda word: WORDS[word] / N #"Probability of `word`."
    
    correction = lambda word: max(candidates(word), key=P) #"Most probable
    return WORDS,correction

WORDS,CORRECTION = DICO_ET_CORRECTEUR()


##stopwords #//https://www.ranks.nl/stopwords/french
with open('stp_words_.txt','r') as f:
    STOPWORDS = f.read()

##bdd de stemmer
with open("sample_.json",'r',encoding='cp1252') as json_file:
    #json_file.seek(0)
    LISTE = json.load(json_file)
my_stemmer = lambda word: LISTE[word] if word in LISTE else word

def SENTENCE_TO_CORRECT_WORDS(sentence):
    "cette fonction retourne la liste des mots du user"
    print('\n------------pre_process--------\n')
    liste_words = pre_process(sentence)
    print(liste_words)
    print('\n------------correction--------\n')
    liste_words = list(map(CORRECTION,liste_words))
    print(liste_words)
    print('\n------------stemming--------\n')
    liste_words = list(map(my_stemmer,liste_words))
    print(liste_words)
    print('\n------------remove stop-words--------\n')
    liste_words = [elt for elt in liste_words if elt not in STOPWORDS]
    print(liste_words)
    print('\n-------------------------------------\n')
    return liste_words
```




---
##Test: SENTENCE_TO_CORRECT_WORDS
---

```
SENTENCE_TO_CORRECT_WORDS('La PR reste au statut «\xa0Approuve(e)\xa0» et il n’y a pas de commande\"\'')
```



---
##Output
---

```
------------pre_process--------
['reste', 'statut', 'approuve', 'n’y', 'pas', 'commande']

------------correction--------
['reste', 'statut', 'approuve', 'n’y', 'pas', 'commande']

------------stemming--------
['rester', 'statut', 'approuver', 'n’y', 'pas', 'commander']

------------remove stop-words--------
['rester', 'statut', 'approuver', 'n’y', 'commander']

-------------------------------------
['rester', 'statut', 'approuver', 'n’y', 'commander']
```








---
#**Create dataset**
---

```
def open_file(textdir):
  found = False
  try:texte = open(textdir,'r',encoding="utf-8").read();found=True
  except:pass
  try: texte = open(textdir,'r').read();found=True 
  except: pass
  if not found:
    texte = open(textdir,'r',encoding='cp1252').read();found=True
  return  texte
def add_col(df_news,titre,keywords):
  return df_news.append(dict(zip(df_news.columns,[titre, keywords])), ignore_index=True)

liste_pb = [elt for elt in open_file('liste_pb_.txt').split('\n') if elt]
df_new = df_news.drop(df_news.index)
for i,titre in enumerate(liste_pb):
  keywords = ','.join(SENTENCE_TO_CORRECT_WORDS(titre))
  df_new = add_col(df_new,titre,keywords)
df_new.head()

```
---
##Output
---
```
 	Subject 	Clean_Keyword
0 	Message d'erreur : "Le fournisseur ARIBA n'exi... 	message,erreur,fournisseur,ariba,exister
1 	Message d'erreur : "Commande d’article non aut... 	message,erreur,commander,d’article,autoriser,otp
2 	Message d'erreur : "Statut utilisateur FERM ac... 	message,erreur,statut,utilisateur,ferm,actif,otp
3 	Message d'erreur : "Statut systeme TCLO actif ... 	message,erreur,statut,systeme,tclo,actif,ord
4 	Message d'erreur "___ Cost center change could... 	message,erreur,cost,center,changer,could,not,e...
5 	Messaeg d'erreur "___ OTP change could not be ... 	messaeg,erreur,otp,changer,could,not,effected
6 	Messaeg d'erreur "Entrez Centre de couts" 	messaeg,erreur,entrer,centrer,cout
7 	Message d'erreur "Indiquez une seule imputatio... 	message,erreur,indiquer,seul,imputation,statis...
8 	Message d'erreur "Imputations CO ont des centr... 	message,erreur,imputation,centrer,profit
9 	Message d'erreur "Poste ___ Ordre ___ depassem... 	message,erreur,poster,ordre,depassement,budget
10 	Message d'erreur "Entrez une quantite de comma... 	message,erreur,entrer,quantite,commander
11 	Message d'erreur "Indiquez la quantite" 	message,erreur,indiquer,quantite
12 	Message d'erreur "Le prix net doit etre superi... 	message,erreur,prix,net,superieur
...
...
...
```











# **2: trouver la meilleure phrase dans une liste de phrase**
---
##fonction: cosine_similarity_T
---


```
from sklearn.feature_extraction.text import TfidfVectorizer
import operator
import numpy as np 
import time


def init(df_news):
  ## Create Vocabulary
  vocabulary = set()
  for doc in df_news.Clean_Keyword:
      vocabulary.update(doc.split(','))
  vocabulary = list(vocabulary)# Intializating the tfIdf model
  tfidf = TfidfVectorizer(vocabulary=vocabulary)# Fit the TfIdf model
  tfidf.fit(df_news.Clean_Keyword)# Transform the TfIdf model
  tfidf_tran=tfidf.transform(df_news.Clean_Keyword)
  globals()['vocabulary'],globals()['tfidf'],globals()['tfidf_tran'] = vocabulary,tfidf,tfidf_tran

def gen_vector_T(tokens,df_news,vocabulary,tfidf,tfidf_tran):
  Q = np.zeros((len(vocabulary)))    
  x= tfidf.transform(tokens)
  #print(tokens[0].split(','))
  #print(keywords)
  for token in tokens[0].split(','):
      
      try:
          ind = vocabulary.index(token)
          Q[ind]  = x[0, tfidf.vocabulary_[token]]
          print(token,':',ind)
      except:
          print(token,':','not found')
          pass
  return Q
def cosine_sim(a, b):
    if not np.linalg.norm(a) and not np.linalg.norm(b): return -3
    if not np.linalg.norm(a):return -1
    if not np.linalg.norm(b):return -2
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim   
def cosine_similarity_T(k, query,df_news,vocabulary=None,tfidf=None,tfidf_tran=None,mine=True):
    try:
      vocabulary = globals()['vocabulary']
      tfidf = globals()['tfidf']
      tfidf_tran = globals()['tfidf_tran']
    except:
      print('up exception')
      init(df_news)
    q_df = pd.DataFrame(columns=['q_clean'])
    if mine:q_df.loc[0,'q_clean'] =','.join(SENTENCE_TO_CORRECT_WORDS(query))
    else:q_df.loc[0,'q_clean'] = wordLemmatizer_(query)
    
    
    print('\n\n---q_df');print(q_df)
    
    print('\n\n')
    d_cosines = []
    query_vector = gen_vector_T(q_df['q_clean'],df_news,vocabulary,tfidf,tfidf_tran )
    for d in tfidf_tran.A:
        d_cosines.append(cosine_sim(query_vector, d ))
                    
    out = np.array(d_cosines).argsort()[-k:][::-1]
    #print("")
    d_cosines.sort()
    a = pd.DataFrame()
    for i,index in enumerate(out):
        a.loc[i,'index'] = str(index)
        a.loc[i,'Subject'] = df_news['Subject'][index]
    for j,simScore in enumerate(d_cosines[-k:][::-1]):
        a.loc[j,'Score'] = simScore
    return a


def test(data,sentence,init_=False,mine=True):
  if not init_:
    deb = time.time();print('\n\n###########')
    init(df_news)
    print('\n###########temps init: ', time.time()-deb)
  deb = time.time();print('\n\n###########')
  print(cosine_similarity_T(10, sentence,df_news))
  print('\n###########temps methode 1: ', time.time()-deb)
```



---
##Test: cosine_similarity_T
---

```
sentence = 'Message d\'erreur \"La qte livree est differente de la qte facturee ; fonction impossible"'
sentence = 'erreur de conversion'
sentence = 'message d\'erreur'
sentence = "le fournisseur MDM n'existe pas"
sentence = "groupe d'acheteurs non défini"
sentence = "UO4"
init(df_new) 

cosine_similarity_T(10,sentence,df_new )
```



---
###Output
---


```
------------pre_process--------
['fournisseur', 'mdm', 'existe', 'pas']

------------correction--------
['fournisseur', 'mdm', 'existe', 'pas']

------------stemming--------
['fournisseur', 'mdm', 'exister', 'pas']

------------remove stop-words--------
['fournisseur', 'mdm', 'exister']

-------------------------------------

	index 	Subject 	Score
0 	19 	Message d'erreur "Le fournisseur MDM___ n’exis... 	0.781490
1 	0 	Message d'erreur : "Le fournisseur ARIBA n'exi... 	0.600296
2 	20 	Message d'erreur "Le fournisseur MDM___ est bl... 	0.587467
3 	14 	Message d'erreur "Le centre de profit __ n'exi... 	0.236420
4 	33 	Message d'erreur "Il existe des factures pour ... 	0.214371
5 	53 	Message d'erreur "Fournisseur non present dans... 	0.142208
6 	18 	Message d'erreur "Validation ___ : le compte _... 	0.000000
7 	30 	Message d'erreur "Renseigner correctement le d... 	0.000000
8 	29 	Message d'erreur "Article ___ non gere dans la... 	0.000000
9 	28 	Message d'erreur "Fonctions oblig. Suivantes n... 	0.000000
```




```

```
