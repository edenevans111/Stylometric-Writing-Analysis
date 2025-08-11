from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

def word_count(string):
    return len(word_tokenize(string))

def unique_word_count(string):
    string = string.lower()
    all_words = word_tokenize(string)
    unique_words = []
    for word in all_words:
        if word.lower() not in unique_words:
            unique_words.append(word)
    return len(unique_words)

def character_count(string):
    return len(string)

def avg_word_length(char_count, word_count):
    return char_count/word_count

def hapax_legomenon_rate(unique_word_count, word_count):
    return unique_word_count/word_count

def unique_proportion(unique, words):
    return unique/words

def sentence_count(string):
    return len(sent_tokenize(string))

def num_stop_words(string):
    stop_words = set(stopwords.word("english"))
    all_stops = []
    for word in string:
        if word.casefold() in stop_words:
            all_stops.append(word)
    return len(all_stops)



def main():
    exampleString = "This is an example to show us all how this works"
    print(word_tokenize(exampleString.lower()))
    print(unique_word_count(exampleString))


if __name__ == '__main__':
    main()