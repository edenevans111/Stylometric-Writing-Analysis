from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

exampleString = "This is an example string to see if I can figure out how to do this"

word_tokenize(exampleString)


def word_count(string):
    # Obviously this one is for counting the number of words
    return len(word_tokenize(string))

def sentence_count(string):
    return len(sent_tokenize(string))

def unique_word_count(string):
    pass

def unique_prop(unique, words):
    return unique/words

def num_stop_words(string):
    stop_words = set(stopwords.word("english"))
    all_stops = []
    for word in string:
        if word.casefold() in stop_words:
            all_stops.append(word)
    return len(all_stops)




def main():
    exampleString = "This is an example to show us all how this works"
    print(word_count(exampleString))


if __name__ == '__main__':
    main()