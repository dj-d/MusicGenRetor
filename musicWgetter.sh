#!/bin/bash

BASE_URL='https://freemusicarchive.org/genre/'
GENRES='Blues Electronic Classical Pop Rock Jazz'

mkdir -p "songs"
cd songs

for GENRE in $GENRES; do
    echo "Crawling: ${GENRE} "
    CRAWLING_URL=${BASE_URL}${GENRE}
    echo "URL: ${CRAWLING_URL}"

    mkdir -p $GENRE && cd $GENRE

    for (( PAGE_NUMBER=1 ; i <= 12; i++ )) ; do
        PAGE_PARAMS="?sort=track_date_published&d=1&page=${PAGE_NUMBER}"
        PAGE_URL=${CRAWLING_URL}${PAGE_PARAMS}

        wget $PAGE_URL -q -O ${GENRE}_${PAGE_NUMBER}

        grep -Po '(?<=href=")[^"]*' ${GENRE}_${PAGE_NUMBER} | grep '/music' | parallel --gnu "wget -A '.mp3' {} "
        echo -e "\n"
    done

    cd ..
done
