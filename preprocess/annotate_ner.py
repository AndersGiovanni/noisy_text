__author__ = "bplank"
"""
Use Space (lg) model to extract entities
"""
import pandas as pd
import spacy
import sys

#nlp = spacy.load("en_core_web_sm") # used for prototyping
nlp = spacy.load("en_core_web_lg") # use large model


def normalize(text):
    print("Text", text)
    # remove httpurl as it gets tagged as person
    if "httpurl" in text:
        text = text.replace("httpurl", "")
    return text

def get_entities(texts):
    texts = [normalize(text) for text in texts]
    for doc in nlp.pipe(texts, disable=["tagger","parser"]):
        #print([(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents])
        return " ".join([ent.label_ for ent in doc.ents])
        #for token in doc:
        #    print(token.text, token.pos_)

def get_entities_details(texts):
    texts = [normalize(text) for text in texts]
    for doc in nlp.pipe(texts, disable=["tagger","parser"]):
        ent_texts = ["/".join([ent.text.replace(" ","_"), ent.label_]) for ent in doc.ents]
        return " ".join(ent_texts)

output = get_entities(["official death toll from #covid19 in the united kingdom is now greater than: germany + poland + switzerland + austria + portugal + greece + sweden + finland + norway + ireland... combined. uk: 67.5 million (233 dead) above group: 185 million (230 dead) httpurl"])
print(output)
#exit()
#     print(ent.text, ent.start_char, ent.end_char, ent.label_)
df1 = pd.read_csv(sys.argv[1], sep="\t")
df1["Entities"] = df1["Text"].apply(lambda x: get_entities([x]))
df1["Entities_Details"] = df1["Text"].apply(lambda x: get_entities_details([x]))

df1.to_csv(sys.argv[2],  index=False,  sep='\t')