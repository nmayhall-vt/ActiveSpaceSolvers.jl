#!/bin/sh

tmp2=""

cp $1 README.md

tmp=$(git config --get remote.origin.url)
final=${tmp/".git/"$tmp2}


gitrepo="{{ github.repository }}"

sed -i '' "s|${gitrepo}|${final}|g" README.md

gituser="{{ github.user }}"

tmp="https://github.com/"
codecov=${final/$tmp/$tmp2}

sed -i '' "s|${gituser}|${codecov}|g" README.md

git add README.md
