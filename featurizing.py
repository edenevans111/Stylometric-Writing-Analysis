from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import string
import csv

# lexical features
def word_count(sentence):
    return len(word_tokenize(sentence))

def unique_word_count(sentence):
    sentence = sentence.lower()
    all_words = word_tokenize(sentence)
    unique_words = []
    for word in all_words:
        if word.lower() not in unique_words:
            unique_words.append(word)
    return len(unique_words)

def character_count(sentence):
    return len(sentence)

def avg_word_length(char_count, word_count):
    return char_count / word_count

def hapax_legomenon_rate(sentence):
    tokens = word_tokenize(sentence.lower())
    word_count = word_count(sentence)
    fdist = FreqDist(tokens)
    hapaxes = fdist.hapaxes()
    return len(hapaxes) / word_count
    

def unique_proportion(unique, words):
    return unique/words

# syntax features
def sentence_count(string):
    return len(sent_tokenize(string))

def average_sent_length(sent_count, word_count):
    return word_count/sent_count

def punctuation_count(sentence):
    word_tokens = word_tokenize(sentence)
    punctuation_marks = set(string.punctuation)
    # I know that I have a good set of punctuation marks here now
    punctuation_count = 0
    for token in word_tokens:
        if token in punctuation_marks:
            punctuation_count += 1
    return punctuation_count


def stop_words_count(sentence):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(sentence)
    stop_count = 0
    for token in word_tokens:
        if token in stop_words:
            stop_count += 1
    return stop_count

def question_count(sentence):
    tokens = word_tokenize(sentence)
    question_count = 0
    for token in tokens:
        if token == '?':
            question_count += 1
    return question_count

# Named Entity features
def first_person_pronouns(sentence):
    tokens = word_tokenize(sentence)
    count = 0
    first_person_list = ['I', 'me', 'mine', 'myself', 'we', 'us', 'our', 'ourselves']
    for token in tokens:
        if token in first_person_list:
            count += 1
    return count

def list_of_features(sentence):
    features = []
    # Lexical features
    words = word_count(sentence)
    unique_words = unique_word_count(sentence)
    characters = character_count(sentence)
    avg_word_lengths = avg_word_length(characters, words)
    hapax_legomenons = hapax_legomenon_rate(sentence)
    # syntax features
    sentences = sentence_count(sentence)
    average_sent_lengths = average_sent_length(sentences, words)
    punctuations = punctuation_count(sentence)
    stop_words = stop_words_count(sentence)
    questions = question_count(sentence)
    first_persons = first_person_pronouns(sentence)

    features.append(words)
    features.append(unique_words)
    features.append(characters)
    features.append(avg_word_lengths)
    features.append(hapax_legomenons)

    features.append(sentences, average_sent_lengths, punctuations, stop_words, questions, first_persons)
    
    return features


## Things that still need to be figured out:
# Emotion word count
# FleschReadingEase
# Contraction Count




def main():

    with open('sample_reasoning_turns_wait.csv') as infile:
        csvreader = csv.reader(infile)
        next(csvreader)
        next_sentence = next(csvreader)
        features = list_of_features(next_sentence[0])
        print(features)

        


if __name__ == '__main__':
    main()