import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.corpus import cmudict
import string
import csv
import pandas as pd
import numpy as np
import spacy
from spacy_syllables import SpacySyllables

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
    words = word_count(sentence)
    fdist = FreqDist(tokens)
    hapaxes = fdist.hapaxes()
    return len(hapaxes) / words
    

def unique_proportion(unique, words):
    return unique/words

# syntax features
def sentence_count(sentence):
    return len(sent_tokenize(sentence))

def average_sent_length(sent_count, word_count):
    return word_count/sent_count

def punctuation_count(sentence):
    word_tokens = word_tokenize(sentence)
    punctuation_marks = set(string.punctuation)
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


def count_syllable(sentence):
    syllable_count = 0
    vowels = ['a', 'e', 'i', 'o', 'u', 'y']
    tokens = word_tokenize(sentence)
    for token in tokens:
        count = 0
        for character in token:
            if character in vowels:
                syllable_count += 1
                if count != 0:
                    if token[count - 1] in vowels:
                        syllable_count -= 1
            count += 1
        if token[len(token) - 1] == 'e':
            syllable_count -= 1
        if len(token) == 1:
            syllable_count += 1
    return syllable_count

def count_syllables_nltk(sentence):
    d = cmudict.dict()
    sentence.lower() 
    tokens = word_tokenize(sentence)
    syllable_count = 0
    for token in tokens:
        if token in d:
            syllable_count += max([len([y for y in x if y[-1].isdigit()]) for x in d[token]])
        else:
            syllable_count += 0
    return syllable_count

def count_syllables_spacy(sentence):
    nlp = spacy.load("en_core_web_sm")
    # I don't know that I really need this line:
    nlp.add_pipe("syllables", after="tagger")
    doc = nlp(sentence)
    syllable_count = 0
    for token in doc:
        syllable_count += token._.syllables_count
    return syllable_count

def flesch_score(sentence):
    words = word_count(sentence)
    sentences = sentence_count(sentence)
    avg_sent_length = average_sent_length(sentences, words)
    syllables = count_syllables_spacy(sentence)
    average_syllables_per_word = syllables / word_count(sentence)
    return 206.835 - (1.015 * avg_sent_length) - (84.6 * average_syllables_per_word)

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
    flesch_reading_score = flesch_score(sentence)

    features.append(words)
    features.append(unique_words)
    features.append(characters)
    features.append(avg_word_lengths)
    features.append(hapax_legomenons)

    features.append(sentences)
    features.append(average_sent_lengths)
    features.append(punctuations)
    features.append(stop_words)
    features.append(questions)
    features.append(first_persons)
    features.append(flesch_reading_score)
    
    return features

def main():
    df = pd.DataFrame(columns=['Word Count', 'Unique Words', 'Characters', 'Average Word Lengths', 
                               'Hapax Legomenons', 'Sentence Count', 'Average Sentence Length', 
                               'Punctuation Marks', 'Stop Words', 'Questions', 'First Person Pronouns', 
                               'Flesch Reading Score', 'Role'])

    with open('sample_reasoning_turns_wait_roles copy.csv') as infile:
        csvreader = csv.reader(infile)
        next(csvreader)
        for row in csvreader:
            features = list_of_features(row[0])
            features.append(row[1])
            df.loc[len(df)] = features

    print(df.head())
    df.to_csv('stylometric_features.csv')

        


if __name__ == '__main__':
    main()