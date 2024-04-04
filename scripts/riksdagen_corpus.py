import argparse

import numpy as np
import pandas as pd
import progressbar
from lxml import etree
from numpy import nanmax, nanmin
from pyparlaclarin.read import paragraph_iterator, speech_iterator, speeches_with_name
from pyriksdagen.utils import protocol_iterators


def speech_iterator(root):
    """
    Convert Parla-Clarin XML to an iterator of of concatenated speeches and speaker ids.
    Speech segments are concatenated unless a new speaker appears (ignoring any notes).

    Args:
        root: Parla-Clarin document root, as an lxml tree root.
    """
    us = root.findall(".//{http://www.tei-c.org/ns/1.0}u")
    protocol_id = root.get("{http://www.w3.org/XML/1998/namespace}id", "")
    # Find all 'docDate' elements
    docdates = root.findall(".//{http://www.tei-c.org/ns/1.0}docDate")
    if len(us) == 0:
        return None

    speaker_id_old = us[0].attrib.get("who", "")
    dates = [docdate.get("when", "") for docdate in docdates]

    speech = []
    for u in us:
        # TODO: Search for speech_id in notes. This is wrong.
        speech_id = u.get("{http://www.w3.org/XML/1998/namespace}id", "")  # WRONG
        for text in u.itertext():
            speaker_id = u.attrib.get("who", "")
            if speaker_id != speaker_id_old:
                yield (
                    {
                        "text": " ".join(speech),
                        "speaker_id": speaker_id,
                        "speech_id": speech_id,
                        "protocol_id": protocol_id,
                        "date": dates,
                    }
                )
                speech = []
            speech.extend(text.split())
            speaker_id_old = speaker_id

    # Yield the last segment
    if speech:
        yield {
            "text": " ".join(speech),
            "speaker_id": speaker_id_old,
            "speech_id": speech_id,
            "date": dates,
        }


mop = pd.read_csv("corpus/metadata/member_of_parliament.csv")
minister = pd.read_csv("corpus/metadata/minister.csv")

# Outer join of mop and minister
mop = pd.concat([mop, minister], ignore_index=True)
# Chaos ensues if we don't convert the dates to datetime objects and want to process them
mop["start"] = pd.to_datetime(mop["start"], format="ISO8601")
mop["end"] = pd.to_datetime(mop["end"], format="ISO8601")

name = pd.read_csv("corpus/metadata/name.csv")
name = name[name["primary_name"]][["swerik_id", "name"]]
person = pd.read_csv("corpus/metadata/person.csv")
party_affiliation = pd.read_csv("corpus/metadata/party_affiliation.csv")
party_affiliation["start"] = pd.to_datetime(party_affiliation["start"], format="ISO8601")
party_affiliation["end"] = pd.to_datetime(party_affiliation["end"], format="ISO8601")
# Combine rows of people who served for multiple congressional terms
party_affiliation = (
    party_affiliation.groupby("swerik_id")
    .agg(
        {
            "start": lambda x: nanmin(x) if not x.isnull().all() else pd.NaT,
            "end": lambda x: nanmax(x) if not x.isnull().all() else pd.NaT,
            "party": lambda x: ", ".join(set(x.dropna())),
            "party_id": lambda x: ", ".join(set(x.dropna())),
        }
    )
    .reset_index()
)
party_1991 = party_affiliation[
    (party_affiliation["start"] < "1992-01-01") & (party_affiliation["end"] >= "1991-01-01")
]

# Combine rows of people who served for multiple congressional terms
mop = (
    mop.groupby("swerik_id")
    .agg(
        {
            # NaT if all values are NaT otherwise min
            "start": lambda x: nanmin(x) if not x.isnull().all() else pd.NaT,
            "end": lambda x: nanmax(x) if not x.isnull().all() else pd.NaT,
            "district": lambda x: ", ".join(set(x.dropna())),
            "role": lambda x: ", ".join(set(x.dropna())),
        }
    )
    .reset_index()
)

# We merge mandate periods of the MOPs with the names of the MOPs
mop = mop.merge(name, on="swerik_id", how="left")
# Let's also add person-level metadata, such as birth year and gender
mop = mop.merge(person, on="swerik_id", how="left")
mop = mop.merge(party_affiliation, on="swerik_id", how="left", suffixes=("", "_party"))


# We need a parser for reading in XML data
parser = etree.XMLParser(remove_blank_text=True)
protocols = list(protocol_iterators("corpus/protocols", start=1990, end=1992))

speeches = []  # 24764
for protocol in progressbar.progressbar(protocols):
    root = etree.parse(protocol, parser).getroot()
    for speech in speech_iterator(root):
        speeches.append(speech)

df_speeches = pd.DataFrame(speeches)
df_speeches = df_speeches.merge(mop, left_on="speaker_id", right_on="swerik_id", how="left")

# Debates 1991 are held in a single day, so we can just take the first date
df_speeches["date"] = df_speeches["date"].apply(lambda x: x[0])

df_speeches.to_parquet("data/riksdagen_speeches.parquet", index=False)
