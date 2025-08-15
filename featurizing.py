import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.corpus import cmudict
import string
import csv
import pandas as pd
import numpy as np

# lexical features
def word_count(sentence_tokens):
    return len(sentence_tokens)

def unique_word_count(sentence_tokens):
    unique_words = []
    for word in sentence_tokens:
        if word.lower() not in unique_words:
            unique_words.append(word)
    return len(unique_words)

def character_count(sentence):
    return len(sentence)

def avg_word_length(char_count, word_count):
    return char_count / word_count

def unique_proportion(unique, words):
    return unique/words

def hapax_legomenon_rate(sentence, sentence_tokens):
    words = word_count(sentence)
    fdist = FreqDist(sentence_tokens)
    hapaxes = fdist.hapaxes()
    return len(hapaxes) / words

# syntax features
def sentence_count(sentence):
    return len(sent_tokenize(sentence))

def average_sent_length(sent_count, word_count):
    return word_count/sent_count

def punctuation_count(sentence_tokens):
    punctuation_marks = set(string.punctuation)
    # I know that I have a good set of punctuation marks here now
    punctuation_count = 0
    for token in sentence_tokens:
        if token in punctuation_marks:
            punctuation_count += 1
    return punctuation_count


def stop_words_count(sentence_tokens):
    stop_words = set(stopwords.words("english"))
    stop_count = 0
    for token in sentence_tokens:
        if token in stop_words:
            stop_count += 1
    return stop_count

def question_count(sentence_tokens):
    question_count = 0
    for token in sentence_tokens:
        if token == '?':
            question_count += 1
    return question_count

def exclamations_count(sentence_tokens):
    exclamation_count = 0
    for token in sentence_tokens:
        if token == '!':
            exclamation_count += 1
    return exclamation_count

def contractions_count(sentence_tokens):
    contractions = ["'s", "'re", "'ve", "'ll", "'d", "'t", "'m", "'n", "'bout", "'cause", "'ya", "'all", "'em", "'gain", "'cept", "'fore", "'round", "'till", "'neath", "'pon", "'neath", "'twixt", "'tween", "'ight", "'ere", "'am", "'tis", "'twas", "'twere", "'twill"]
    contraction_count = 0
    for token in sentence_tokens:
        if token in contractions:
            contraction_count += 1
    return contraction_count

# Readability features
def count_syllables_nltk(word):
    # nltk.download('cmudict')
    d = cmudict.dict()
    if word in d:
        return max([len([y for y in x if y[-1].isdigit()]) for x in d[word]])
    return 0

def readability_features(sentence, sentence_tokens, words, sentences):
    avg_sent_length = average_sent_length(sentences, words)
    syllables = 0
    complex_word_count = 0
    for token in sentence_tokens:
        syllable_count = count_syllables_nltk(token)
        syllables += syllable_count
        if syllable_count >= 3:
            complex_word_count += 1
    average_syllables_per_word = syllables / words
    flesch = 206.835 - (1.015 * avg_sent_length) - (84.6 * average_syllables_per_word)
    if sentences == 0 or words == 0:
        gunning_fog = 0
    else:
        gunning_fog = 0.4 * ((words/sentences) + 100 * (complex_word_count/words))
    return flesch, gunning_fog

# Named Entity features
def first_person_pronouns(sentence_tokens):
    count = 0
    first_person_list = ['I', 'me', 'mine', 'myself', 'we', 'us', 'our', 'ourselves']
    for token in sentence_tokens:
        if token in first_person_list:
            count += 1
    return count

def direct_addresses_count(sentence_tokens):
    count = 0
    direct_address_list = ['you', 'your', 'yours', 'yourself', 'yourselves', 'hey', 'hi', 'hello']
    for token in sentence_tokens:
        if token in direct_address_list:
            count += 1
    return count

def list_of_features(sentence):
    sentence_tokens = word_tokenize(sentence.lower())
    features = []
    # Lexical features
    words = word_count(sentence_tokens)
    unique_words = unique_word_count(sentence_tokens)
    characters = character_count(sentence)
    avg_word_lengths = avg_word_length(characters, words)
    unique_prop = unique_proportion(unique_words, words)
    hapax_legomenons = hapax_legomenon_rate(sentence, sentence_tokens)
    # syntax features
    sentences = sentence_count(sentence)
    average_sent_lengths = average_sent_length(sentences, words)
    punctuations = punctuation_count(sentence_tokens)
    stop_words = stop_words_count(sentence_tokens)
    questions = question_count(sentence_tokens)
    exclamations = exclamations_count(sentence_tokens)
    contractions = contractions_count(sentence_tokens)
    # Readability features
    flesch_reading_score, gunning_fog = readability_features(sentence, sentence_tokens, words, sentences)
    # Named Entity features
    first_persons = first_person_pronouns(sentence_tokens)
    direct_addresses = direct_addresses_count(sentence_tokens)

    features.append(words)
    features.append(unique_words)
    features.append(characters)
    features.append(avg_word_lengths)
    features.append(unique_prop)
    features.append(hapax_legomenons)

    features.append(sentences)
    features.append(average_sent_lengths)
    features.append(punctuations)
    features.append(stop_words)
    features.append(questions)
    features.append(exclamations)
    features.append(contractions)

    features.append(flesch_reading_score)
    features.append(gunning_fog)

    features.append(first_persons)
    features.append(direct_addresses)
    
    return features


## Things that still need to be figured out:
# AbstractNounCount
# ComplexVerbCount
# SophisticatedAdjectiveCount
# AdverbCount
# ComplexSentenceCount
# Emotion word count
# Polarity
# Subjectivity
# VaderCompound
# Person Entities
# Date Entities
# Bigram uniqueness
# Trigram uniqueness
# Syntax variety

def main():
    # Need to make a pandas dataframe
    df = pd.DataFrame(columns=['Word Count', 'Unique Words', 'Characters', 'Average Word Lengths', 'Unique Proportion',
                               'Hapax Legomenons', 'Sentence Count', 'Average Sentence Length',
                               'Punctuation Marks', 'Stop Words', 'Questions', 'Exclamations', 'Contractions',
                               'Flesch Reading Score', 'Gunning Fog', 'First Person Pronouns', 'Direct Addresses', 'Role'])

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