#!/bin/bash

BASE_URL='https://freemusicarchive.org/'
GENRE_ENDPOINT='genre/'
MUSIC_ENDPOINT='music/'

GENRES='Blues Electronic Classical Pop Rock Jazz'

mkdir -p "songs"
cd songs

for GENRE in $GENRES; do
    echo "Crawling: ${GENRE} "
    CRAWLING_URL=${BASE_URL}${GENRE_ENDPOINT}${GENRE}
    echo "URL: ${CRAWLING_URL}"

    mkdir -p $GENRE && cd $GENRE

    for PAGE_NUMBER in {1..1} ; do
        PAGE_PARAMS="?sort=track_date_published&d=1&page=${PAGE_NUMBER}"
        PAGE_URL=${CRAWLING_URL}${PAGE_PARAMS}

        wget $PAGE_URL -q -O ${GENRE}_${PAGE_NUMBER}

	BASE_URL_ESCAPED=$(${BASE_URL@Q}${MUSIC_ENDPOINT@Q} | tr -d \')
	URL_REGEX='(?<=href=")'${BASE_URL_ESCAPED}'[^"]*'

        grep -Po ${URL_REGEX} ${GENRE}_${PAGE_NUMBER} | parallel --gnu "wget -A '.mp3' {} "

        echo -e "\n"
    done

    rm ${GENRE}_* *.tmp*
    cd ..
done
