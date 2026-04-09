#!/bin/bash

set -euo pipefail

# Example: download-wikipedia.sh "Python_(programming_language)"
ARTICLE_TITLE=${1:?}
WIKI_COUNTRY=${2:-en}
WIKI_HOST=${3:-wikipedia.org}


curl -s "https://${WIKI_COUNTRY}.${WIKI_HOST}/w/api.php?action=query&prop=revisions&titles=${ARTICLE_TITLE}&rvslots=main&rvprop=content&formatversion=2&format=json" \
    | jq -r '.query.pages[0].revisions[0].slots.main.content' \
    | pandoc -f mediawiki -t markdown
