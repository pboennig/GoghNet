#!/bin/sh
# NOTE: This script is only to be used for purposes that comply with copyright laws.

# Get HTML of page from user's input, get all of the image links, and make sure URLs have HTTPs
curl $1 | grep -E "(https?:)?//[^/\s]+/\S+\.(jpg|png|gif)" -o | sed -E "s/^(https?)?\/\//https\:\/\//g" > urls.txt

# Get full-res URLs instead of thumbnails and re-saving urls.txt
sed -Ei "s/\/thumb//g; s/\/[0-9]+px-.+\.(jpg|png)$//g" urls.txt

# Downloading Images
wget -i urls.txt -P downloads/
