# SearchEngine

<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph="">In this post, we will be building a <strong class="id iq">semantic documents search engine</strong></p>

##Prerequistes
*   Python >=3.7
*   NLTK
*   Pandas
*   Scikit-learn

##Prerequistes
```
import re, json
import unicodedata, string
import time
import operator
import numpy as np 
import pandas as pd
from collections import Counter
```
```
from collections import defaultdict
import nltk 
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
```
```
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
```

##data
Files used in the notebook are stored in the folder data


# **1: Créer les keywords à partir d'une phrase en se basant sur les mots d'un dictionnaire et un corpus de texte en passant par la tokenization, la correction, la lemmatization et le removeStopWords**




--- 
##preprocessing
--- 
```
def get_dico():
    textdir = "liste.de.mots.francais.frgut_.txt"
    try:DICO = open(textdir,'r',encoding="utf-8").read()
    except: DICO = open(textdir,'r').read()
    
    return DICO


def remove_accents(input_str):
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
```
--- 
##correction des mots
--- 

```
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
```
--- 
##stopwords et stemming(premier exemple)
--- 

```

##stopwords #//https://www.ranks.nl/stopwords/french
with open('stp_words_.txt','r') as f:
    STOPWORDS = f.read()

##bdd de stemmer
with open("sample_.json",'r',encoding='cp1252') as json_file:
    #json_file.seek(0)
    LISTE = json.load(json_file)
my_stemmer = lambda word: LISTE[word] if word in LISTE else word
```

---
##fonction: SENTENCE_TO_CORRECT_WORDS
--- 
```
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
##**Create dataset**
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
 	                   Subject 	                                  Clean_Keyword
0 	Message d'erreur : "Le fournisseur ARIBA n'exi... 	message,erreur,fournisseur,aria,exister
1 	Message d'erreur : "Commande d’article non aut... 	message,erreur,commander,article,autoriser,oto
2 	Message d'erreur : "Statut utilisateur FERM ac... 	message,erreur,statut,utilisateur,actif,oto
3 	Message d'erreur : "Statut systeme TCLO actif ... 	message,erreur,statut,systeme,col,actif,nord
4 	Message d'erreur "___ Cost center change could... 	message,erreur,coat,centrer,changer,cold,affecter
5 	Messaeg d'erreur "___ OTP change could not be ... 	message,erreur,otp,changer,cold,affecter
6 	Messaeg d'erreur "Entrez Centre de couts" 	        message,erreur,entrer,centrer,cout
7 	Message d'erreur "Indiquez une seule imputatio... 	message,erreur,indiquer,imputation,statistique
8 	Message d'erreur "Imputations CO ont des centr... 	message,erreur,imputation,centrer,profit
9 	Message d'erreur "Poste ___ Ordre ___ depassem... 	message,erreur,poster,ordre,depassement,budget
10 	Message d'erreur "Entrez une quantite de comma... 	message,erreur,entrer,quantite,commander
11 	Message d'erreur "Indiquez la quantite" 	        message,erreur,indiquer,quantite
12 	Message d'erreur "Le prix net doit etre superi... 	message,erreur,prix,net,superieur
... 	... 	...
... 	... 	...
... 	... 	...
57 	UO4-5 Commande | Envoi d'une commande manuelle 	uo4,commander,envoi,commander,manuel
58 	UO5-4 Reception | Anomalie workflow 	uo5,reception,anomalie,workflow
59 	UO5-1 Reception | Modification(s) de reception(s) 	uo5,reception,modification,reception
60 	UO5-2 Reception | Annulation(s) de reception(s) 	uo5,reception,annulation,reception
61 	UO5-3 Reception | Forcer la reception 	uo5,reception,forcer,reception
62 	UO3-5 Demande d'achat | Demande de support cre... 	uo3,demander,achat,demander,support,creation
63 	UO3-6 Demande d'achat | Demande de support mod... 	uo3,demander,achat,demander,support,modification
64 	UO3-7 Demande d'achat | Demande de support ann... 	uo3,demander,achat,demander,support,annulation
65 	UO4-2 Commande | Demande de support modificati... 	uo4,commander,demander,support,modification,co...
```



---
##tokenize and stemming(second exemple)
---

```
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
def wordLemmatizer(data,colname):
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    file_clean_k =pd.DataFrame()
    for index,entry in enumerate(data):
        
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if len(word)>1 and word not in stopwords.words('french') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
            # The final processed set of words for each iteration will be stored in 'text_final'
                file_clean_k.loc[index,colname] = str(Final_words)
                file_clean_k.loc[index,colname] = str(Final_words)
                file_clean_k=file_clean_k.replace(to_replace ="\[.", value = '', regex = True)
                file_clean_k=file_clean_k.replace(to_replace ="'", value = '', regex = True)
                file_clean_k=file_clean_k.replace(to_replace =" ", value = '', regex = True)
                file_clean_k=file_clean_k.replace(to_replace ='\]', value = '', regex = True)

    return file_clean_k


def wordLemmatizer_(sentence):
    #prendre une phrase que retourner un str (les mots sont separes par des ,)
    preprocessed_query = preprocessed_query = re.sub("\W+", " ", sentence).strip()
    tokens = word_tokenize(str(preprocessed_query))
    q_df = pd.DataFrame(columns=['q_clean'])
    idx = 0
    colname = 'keyword_final'
    q_df.loc[idx,'q_clean'] =tokens
    print('\n\n---inputtoken');print(q_df.q_clean)
    print('\n\n---outputlemma');print(wordLemmatizer(q_df.q_clean,colname).loc[idx,colname])
    return wordLemmatizer(q_df.q_clean,colname).loc[idx,colname]

```




# **2: trouver la meilleure phrase dans une liste de phrase**
---
##fonction: cosine_similarity_T
---


```
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


```
---
##Create a vector for Query/search keywords
---

```
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
```
---
##Cosine Similarity function
---

```
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
```



---
##Test: cosine_similarity_T
---

```
def test(data,sentence,init_=False,mine=True):
  if not init_:
    deb = time.time();print('\n\n###########')
    init(df_news)
    print('\n###########temps init: ', time.time()-deb)
  deb = time.time();print('\n\n###########')
  print(cosine_similarity_T(10, sentence,df_news))
  print('\n###########temps methode 1: ', time.time()-deb)
sentence = 'Message d\'erreur \"La qte livree est differente de la qte facturee ; fonction impossible"'
sentence = 'erreur de conversion'
sentence = 'message d\'erreur'
sentence = "groupe d'acheteurs non défini"
sentence = "UO4"
sentence = "le fournisseur MDM n'existe pas"
init(df_new) 

cosine_similarity_T(10,sentence,df_new )
```



---
##Output
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

       index 	                 Subject 	                         Score
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
... 	... 	...
```




```

```
