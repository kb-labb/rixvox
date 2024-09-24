# Download text and metadata for modern speeches that are available in Riksdagen Open Data API
mkdir -p data 
mkdir -p data/audio
mkdir -p data/audio/json

wget https://data.riksdagen.se/dataset/anforande/anforande-202324.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-202223.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-202122.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-202021.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201920.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201819.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201718.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201617.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201516.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201415.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201314.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201213.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201112.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-201011.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-200910.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-200809.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-200708.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-200607.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-200506.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-200405.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-200304.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-200203.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-200102.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-200001.json.zip -P data/audio/json
wget https://data.riksdagen.se/dataset/anforande/anforande-19992000.json.zip -P data/audio/json

unzip data/audio/json/anforande-202223.json.zip -d data/audio/json/2023
unzip data/audio/json/anforande-202122.json.zip -d data/audio/json/2022
unzip data/audio/json/anforande-202021.json.zip -d data/audio/json/2021
unzip data/audio/json/anforande-201920.json.zip -d data/audio/json/2020
unzip data/audio/json/anforande-201819.json.zip -d data/audio/json/2019
unzip data/audio/json/anforande-201718.json.zip -d data/audio/json/2018
unzip data/audio/json/anforande-201617.json.zip -d data/audio/json/2017
unzip data/audio/json/anforande-201516.json.zip -d data/audio/json/2016
unzip data/audio/json/anforande-201415.json.zip -d data/audio/json/2015
unzip data/audio/json/anforande-201314.json.zip -d data/audio/json/2014
unzip data/audio/json/anforande-201213.json.zip -d data/audio/json/2013
unzip data/audio/json/anforande-201112.json.zip -d data/audio/json/2012
unzip data/audio/json/anforande-201011.json.zip -d data/audio/json/2011
unzip data/audio/json/anforande-200910.json.zip -d data/audio/json/2010
unzip data/audio/json/anforande-200809.json.zip -d data/audio/json/2009
unzip data/audio/json/anforande-200708.json.zip -d data/audio/json/2008
unzip data/audio/json/anforande-200607.json.zip -d data/audio/json/2007
unzip data/audio/json/anforande-200506.json.zip -d data/audio/json/2006
unzip data/audio/json/anforande-200405.json.zip -d data/audio/json/2005
unzip data/audio/json/anforande-200304.json.zip -d data/audio/json/2004
unzip data/audio/json/anforande-200203.json.zip -d data/audio/json/2003
unzip data/audio/json/anforande-200102.json.zip -d data/audio/json/2002
unzip data/audio/json/anforande-200001.json.zip -d data/audio/json/2001
unzip data/audio/json/anforande-19992000.json.zip -d data/audio/json/2000

# Find json.zip and remove them
find data/audio/json -name "*.json.zip" -exec rm -f {} \;