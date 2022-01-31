import textstat as ts

FRE_INDEX = {
    range(0, 10): "Extremely difficult",
    range(10, 30): "Very difficult",
    range(30, 50): "Difficult",
    range(50, 60): "Fairly difficult",
    range(60, 70): "Plain English",
    range(70, 80): "Fairly easy",
    range(80, 90): "Easy",
    range(90, 100): "very easy",
}

GF_INDEX = {
    17: "College graduate",
    16: "College senior",
    15: "College junior",
    14: "College sophomore",
    13: "College freshman",
    12: "High school senior",
    11: "High school junior",
    10: "High school sophomore",
    9: "High school freshman",
    1: "1st grade",
}.update({i: f"{i}th grade" for i in range(2, 9)})

# Flesch reading-ease


def flesch_reading_ease(sentence):
    """In the Flesch reading-ease test, higher scores indicate material that is easier to read;
    lower numbers mark passages that are more difficult to read."""
    fr = ts.flesch_reading_ease(sentence)
    return fr, FRE_INDEX[int(fr)]


# Flesch-Kincaid grade level
def flesch_kincaid(sentence):
    """Returns the grade required to be able to read the text"""
    fk = ts.flesch_kincaid_grade(sentence)
    return fk, GF_INDEX[int(fk)]


def gunning_fog(sentence):
    """Return the grade required to be able to read the text"""
    gf = ts.gunning_fog(sentence)
    return gf, GF_INDEX[int(gf)]


def automated_readability_index(sentence):
    """Returns the ARI (Automated Readability Index) which outputs a number that approximates the grade level needed to comprehend the text."""
    ari = ts.automated_readability_index(sentence)
    return ari, GF_INDEX[int(ari)]


# sentencizer

# reading time
def reading_time(sentence):
    """returns the time (in seconds) needed for reading the text"""
    return ts.reading_time(sentence)


# syllable count
def syllable_count(sentence):
    """return the count of syllable in the text"""
    return ts.syllable_count(sentence)


# lexicon
def lexicon_count(sentence):
    return ts.lexicon_count(sentence)


def difficult_words(sentence):
    return ts.difficult_words(sentence)


def character_count(sentence):
    return ts.char_count(sentence)


def word_count(sentence):
    return ts.word_count(sentence)
