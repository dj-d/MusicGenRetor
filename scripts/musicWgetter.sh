#!/bin/bash

BASE_URL='https://freemusicarchive.org/'
GENRE_ENDPOINT='genre/'
PATTERN='https://files.freemusicarchive.org/storage-freemusicarchive-org/music/'
BASE_URL_ESCAPED=$(echo ${PATTERN} | sed 's/\//\\\//g')
GENRES='Blues Electronic Classical Pop Rock Jazz'
DIR="$( cd "$( dirname "$0" )" && pwd )"
PAGE_NUMBERS=250

# Create songs directory
cd "${DIR}/.."
mkdir -p "songs" && cd songs

for GENRE in $GENRES ; do
    echo "Crawling: ${GENRE} "
    CRAWLING_URL=${BASE_URL}${GENRE_ENDPOINT}${GENRE}
    echo "URL: ${CRAWLING_URL}"

    mkdir -p $GENRE && cd $GENRE

    BEHIND_PATTERN=$(cat <<EOF
>${GENRE}</a>[[:space:]]</span>[[:space:]]</div>[[:space:]]<span[[:space:]]class="playicn">[[:space:]]<a[[:space:]]href="
EOF
		  )
    BEHIND_PATTERN_ESCAPED=$(echo ${BEHIND_PATTERN} | sed 's/\//\\\//g')
    URL_REGEX='(?<='${BEHIND_PATTERN_ESCAPED}')'${BASE_URL_ESCAPED}'[^"]*'

    for PAGE_NUMBER in $( seq 1 $PAGE_NUMBERS) ; do
        PAGE_PARAMS="?sort=track_date_published&d=1&page=${PAGE_NUMBER}"
        PAGE_URL=${CRAWLING_URL}${PAGE_PARAMS}

	echo ${PAGE_URL}

        wget $PAGE_URL -q -O ${GENRE}_${PAGE_NUMBER}
	grep -zPo ${URL_REGEX} ${GENRE}_${PAGE_NUMBER} | tr "\0" "\n" | xargs -0 -n1 | parallel --gnu "wget -A '.mp3' {}"

        echo -e "\n"
    done

    # Remove HTML pages and tmp trash file
    rm -f ${GENRE}_* *.tmp*

    # Remove duplicated
    rm -f *.1

    cd ..
done
