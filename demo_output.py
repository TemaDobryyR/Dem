import itertools
import pickle
from crf_entity_extractor import CrfEntityExtractor
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

# Loading model
crf_loaded = CrfEntityExtractor()
crf_loaded.load_model('CRF_address_ner')
  
def tokenizer(text):
    
    # Tokenize text into sentences
    
    punkt_param = PunktParameters()
    with open("abbrev_list.pkl", "rb") as fp:
        abbrev_list = pickle.load(fp)
        punkt_param.abbrev_types = set(abbrev_list)
        tokenizer = PunktSentenceTokenizer(punkt_param)
        tokenizer.train(text)
        
    all_sentences = tokenizer.tokenize(text)

    seen = set()
    sentences = []
    for sentence in all_sentences:
        if sentence not in seen:
            seen.add(sentence)
            sentences.append(sentence)
    
    return sentences

def output(text, model = crf_loaded):
    
    # Tokenize text into sentences
    text = tokenizer(text)
    
    # Predict labels
    predicted_labels = [model.predict(sentence) for sentence in text]
    predicted_labels = list(itertools.chain(*predicted_labels))
    
    # Making output
    a = []
    for sentence in text:
        words = word_tokenize(sentence, language='portuguese')
        for word in words:
            a.append(word)
    
    output = dict(zip(a, predicted_labels))
    
    return output

def output_2(text, model = crf_loaded):
    
    # Tokenize text into sentences
    text = tokenizer(text)
    
    # Predict labels
    predicted_labels = [model.predict(sentence) for sentence in text]
    
    # Making output
    a = [sentences for sentences in text]
    
    
    output = dict(zip(a, predicted_labels))
    
    return output