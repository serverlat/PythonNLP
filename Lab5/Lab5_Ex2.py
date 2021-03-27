from collections import Counter
import wikipedia
import spotlight
import requests
from datetime import datetime

person = {
    'types': "DBpedia:Person"
}

hm = wikipedia.WikipediaPage(title="Hayao Miyazaki").content

annotationURL = "https://api.dbpedia-spotlight.org/en/annotate"
annotation = spotlight.annotate(annotationURL, hm, confidence=0.7, support=30, filters=person)

counter = Counter([entry["URI"].split("/")[-1] for entry in annotation]) 
print(counter.keys()) # individual entities

properties = ["dbo%3AbirthPlace", "dbo%3AbirthDate"]

for person in counter:
    DBpedia_URL = f"http://vmdbpedia.informatik.uni-leipzig.de:8080/api/1.0.0/values?entities={person}&property={properties[0]}&property={properties[1]}&pretty=NONE&limit=1&offset=0&key=1234&oldVersion=true"
    r = requests.get(DBpedia_URL, headers={"Accept":"application/json"})
    data = r.json()
    name = person.replace("_", " ")
    try:
        birthDate = data["results"]["bindings"][0]["dbobirthDate"]["value"]
        birthDate = datetime.fromisoformat(birthDate).strftime("%d %B, %Y")
        birthPlace = data["results"]["bindings"][0]["dbobirthPlace"]["value"]
        if "http" in birthPlace:
            birthPlace = birthPlace.split("/")[-1].replace("_", " ")
        print(f"{name} was born in {birthPlace} on {birthDate}.")
    except:
        print(f"The required data for {name} could not be fetched :(")
