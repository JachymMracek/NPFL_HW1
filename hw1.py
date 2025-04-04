import unicodedata
from sacremoses import MosesTokenizer
from collections import Counter
import math
import sys

class Chars:
    def __init__(self):
        self.symbols = set()
    
    def is_symbol_in(self,char):
        if char in self.symbols:
            return True
    
    def is_correct(self,char) -> bool:
        if char.isalpha() and unicodedata.name(char, "").startswith("LATIN"):
            self.symbols.add(char)
            return True
    
        else:
            return False

class Formator:
    def correct_text(filename,tag,SPACE = " "):
        chars = Chars()

        text = ""

        with open(filename,"r" ,encoding='utf-8') as language_file:
                for line in language_file:
                    for char in line:
                        
                        if chars.is_symbol_in(char) or chars.is_correct(char):
                            text += char
                        
                        elif char == SPACE:
                            text += SPACE

        tokens = Formator.__tokenize(text,tag)

        return tokens
    
    def __tokenize(text, tag, COUNT_WORDS=322607):
        mt = MosesTokenizer(lang=tag)
        tokens = mt.tokenize(text)
        
        word_counts = Counter(tokens)
        
        most_common_words = word_counts.most_common()
        
        tokens_available = []
        counted_words = 0
        position = 0
        
        while counted_words < COUNT_WORDS:
            word, count = most_common_words[position]
            
            tokens_available.extend([word] * count)
            counted_words += count
            position += 1
        
        size_in_bytes = sys.getsizeof(tokens_available)

        print(size_in_bytes)
        print(counted_words)
        
        return tokens_available
    
    def get_index(quotient,count_words):
        return int((quotient*count_words) // 1)
    
    def split(tokens,TRAIN_QUOTIENT = 0.8,HELDOUT_QUOTIENT = 0.1):
        count_words = len(tokens)

        train_index = Formator.get_index(TRAIN_QUOTIENT,count_words)
        heldout_index = train_index + Formator.get_index(HELDOUT_QUOTIENT,count_words)

        train = tokens[:train_index + 1]
        heldout = tokens[train_index + 1:heldout_index+1]
        test = tokens[heldout_index+1:]

        return train,heldout,test
    
    def get_ngrams(tokens, n):
        ngrams = []

        for token in tokens:
            token_count = len(token)

            for i in range(token_count - n + 1):
                ngrams.append(token[i:i+n])
                
        return ngrams

class Model:
    N = 3
    def __init__(self):
        self.V = None
        self.best_lambda = None
        self.counter = Counter()
    
    def __report(self,trigrams_count,BEST=5):
        trigrams = [(trigram, count) for trigram, count in self.counter.items() if len(trigram) == 3]

        best_trigrams = sorted(trigrams, key=lambda x: x[1], reverse=True)[:BEST]

        for ranking, (trigram, count) in enumerate(best_trigrams):
            print(str(ranking + 1)  + "th" + " most common trigram is " + trigram + 
                  " with count " + str(count) + " and relative frequence " + str(round(count/trigrams_count,2)))
    
    def fit(self,train_tokens,tokens_heldout):
        ngram_probability = {}
        trigrams_count = None

        for n in range(1,self.N+1):
            ngrams = Formator.get_ngrams(train_tokens,n)
            n_counter = Counter(ngrams)
            self.counter += n_counter

            self.__probability_ngram(ngrams,ngram_probability,n)

            if n == self.N:
                trigrams_count = sum(n_counter.values())
        
        self.__report(trigrams_count)

        trigrams_heldout = Formator.get_ngrams(tokens_heldout,self.N)

        self.best_lambda = self.__get_best_lambda(trigrams_heldout)
    
    def __probability_ngram(self,ngrams,ngram_probability,n):
        for ngram in ngrams:
            if n == 1:
                history = len(self.counter)
                self.V = history
            else:
                history = self.counter[ngram[:-1]]
        
            ngram_probability[ngram] = self.counter[ngram] / history

    def __smoothing_probability(self, ngram, lambda_choosen):

        if ngram not in self.counter:
            count_ngram = 0
        else:
            count_ngram = self.counter[ngram]

        if ngram[:-1] not in self.counter:
            count_history = 0
        else:
            count_history = self.counter[ngram[:-1]]

        return (count_ngram + lambda_choosen) / (count_history + lambda_choosen * self.V)

    def probability_language(self, ngrams_test):
        log_p_language = 0  # GPT Markov chain mi poddíkal na nulu.

        for ngram in ngrams_test:
            prob = self.__smoothing_probability(ngram, self.best_lambda)
            log_p_language += math.log(prob)
            
        return log_p_language
    
    def cross_entropy(self,ngrams,lambda_choosen):
        cross_entropy = 0

        for ngram in ngrams:
            p = self.__smoothing_probability(ngram,lambda_choosen)

            if p < 0 or p > 1:
                cross_entropy = float("inf")
            else:
                cross_entropy += math.log2(p)

        cross_entropy *= -1 / len(ngrams)

        return cross_entropy
    
    def __get_best_lambda(self,ngrams):
        best_lambda = None
        best_cross_entropy = float("inf")

        lambdas = [x * 0.001 for x in range(1, 1000)]

        for lambda_choosen in lambdas:
            cross_entropy = self.cross_entropy(ngrams,lambda_choosen)
        
            if  best_cross_entropy > cross_entropy:
                best_cross_entropy = cross_entropy
                best_lambda = lambda_choosen
        
        return best_lambda

class Language:
    def __init__(self,tag,filename):
        self.model = Model()
        self.tag = tag

        train_tokens,heldout_tokens,self.test_tokens = self.__read_data(tag,filename)

        self.model.fit(train_tokens,heldout_tokens)

        print("Test cross entropy of language " + tag + " is " + str(self.model.cross_entropy(Formator.get_ngrams(self.test_tokens,self.model.N),self.model.best_lambda)))
    
    def __read_data(self,tag,datafilename):
        print("Language informations: ")
        tokens = Formator.correct_text(datafilename,tag)
        return Formator.split(tokens)

class Classificator:
    def predict(languages:list[Language],filename):
        predictions = []

        for language in languages:
            tokens = Formator.correct_text(filename,language.tag)

            ngrams = Formator.get_ngrams(tokens, Model.N)

            p = language.model.probability_language(ngrams)

            predictions.append((language.tag,p))
        
        predictions_sorted = sorted(predictions, key=lambda x: x[1], reverse=True)
        print(predictions_sorted[0])

        return predictions

if __name__ == "__main__":
    input_filename = "unknown.txt"
    filenames = ["english.txt","dutch.txt","italian.txt"]
    tags = ["en","nl","it"]
    languages = []

    for i in range(3):
        languages.append(Language(tags[i],filenames[i]))
    
    # for i in range(3):
       # Classificator.predict(languages,languages[i].test_tokens)
    
    Classificator.predict(languages,input_filename)