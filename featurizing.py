import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.corpus import cmudict
import string
import csv
import pandas as pd
import numpy as np

# helper functions
def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count

def count_syllable(sentence):
    syllable_counts = 0
    tokens = word_tokenize(sentence.lower())
    for token in tokens:
        syllable_counts += syllable_count(token)
    return syllable_counts

# lexical features
def word_count(sentence):
    return len(word_tokenize(sentence))

def unique_word_count(sentence):
    all_words = word_tokenize(sentence.lower())
    unique_words = []
    for word in all_words:
        if word.lower() not in unique_words:
            unique_words.append(word)
    return len(unique_words)

def character_count(sentence):
    return len(sentence)

def avg_word_length(char_count, word_count):
    return char_count / word_count

def unique_proportion(unique, words):
    return unique/words

def hapax_legomenon_rate(sentence):
    tokens = word_tokenize(sentence.lower())
    words = word_count(sentence)
    fdist = FreqDist(tokens)
    hapaxes = fdist.hapaxes()
    return len(hapaxes) / words

# syntax features
def sentence_count(sentence):
    return len(sent_tokenize(sentence))

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
    word_tokens = word_tokenize(sentence.lower())
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

def exclamations_count(sentence):
    tokens = word_tokenize(sentence)
    exclamation_count = 0
    for token in tokens:
        if token == '!':
            exclamation_count += 1
    return exclamation_count

def contractions_count(sentence):
    tokens = word_tokenize(sentence.lower())
    contractions = ["'s", "'re", "'ve", "'ll", "'d", "'t", "'m", "'n", "'bout", "'cause", "'ya", "'all", "'em", "'gain", "'cept", "'fore", "'round", "'till", "'neath", "'pon", "'neath", "'twixt", "'tween", "'ight", "'ere", "'am", "'tis", "'twas", "'twere", "'twill"]
    contraction_count = 0
    for token in tokens:
        if token in contractions:
            contraction_count += 1
    return contraction_count

# Readability features
def count_syllables_nltk(sentence):
    # nltk.download('cmudict')
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

def flesch_score(sentence):
    words = word_count(sentence)
    sentences = sentence_count(sentence)
    avg_sent_length = average_sent_length(sentences, words)
    syllables = count_syllables_nltk(sentence)
    average_syllables_per_word = syllables / word_count(sentence)
    return 206.835 - (1.015 * avg_sent_length) - (84.6 * average_syllables_per_word)

def gunning_fog_index(words, sentences, text):
    # Gunning Fog Index = 0.4 * [(words/sentences) + 100*(complex words/words)]
    # Complex words are those with 3 or more syllables
    if sentences == 0 or words == 0:
        return 0
    word_tokens = word_tokenize(text.lower())
    complex_word_count = 0
    for word in word_tokens:
        syllable_count = syllable_count(word)
        if syllable_count >= 3:
            complex_word_count += 1
    gunning_fog = 0.4 * ((words/sentences) + 100 * (complex_word_count/words))
    return gunning_fog

# Named Entity features
def first_person_pronouns(sentence):
    tokens = word_tokenize(sentence.lower())
    count = 0
    first_person_list = ['I', 'me', 'mine', 'myself', 'we', 'us', 'our', 'ourselves']
    for token in tokens:
        if token in first_person_list:
            count += 1
    return count

def direct_addresses_count(sentence):
    tokens = word_tokenize(sentence.lower())
    count = 0
    direct_address_list = ['you', 'your', 'yours', 'yourself', 'yourselves', 'hey', 'hi', 'hello']
    for token in tokens:
        if token in direct_address_list:
            count += 1
    return count

def list_of_features(sentence):
    features = []
    # Lexical features
    words = word_count(sentence)
    unique_words = unique_word_count(sentence)
    characters = character_count(sentence)
    avg_word_lengths = avg_word_length(characters, words)
    unique_prop = unique_proportion(unique_words, words)
    hapax_legomenons = hapax_legomenon_rate(sentence)
    # syntax features
    sentences = sentence_count(sentence)
    average_sent_lengths = average_sent_length(sentences, words)
    punctuations = punctuation_count(sentence)
    stop_words = stop_words_count(sentence)
    questions = question_count(sentence)
    exclamations = exclamations_count(sentence)
    contractions = contractions_count(sentence)
    # Readability features
    flesch_reading_score = flesch_score(sentence)
    gunning_fog = gunning_fog_index(words, sentences, sentence)
    # Named Entity features
    first_persons = first_person_pronouns(sentence)
    direct_addresses = direct_addresses_count(sentence)

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