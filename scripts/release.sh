#!/bin/bash

set -e

if [[ $(git status -suno) ]]; then
    echo "There are uncommitted changes on this git repository, please commit these and try again."
    exit 1
fi

TAG=$(python -c 'from prior.version import VERSION; print("v" + VERSION)')

read -p "Creating new release for $TAG. Do you want to continue? [Y/n] " prompt

if [[ $prompt == "y" || $prompt == "Y" || $prompt == "yes" || $prompt == "Yes" ]]; then
    python scripts/prepare_changelog.py
    git add CHANGELOG.md
    (git commit -m "$TAG release" || true) && git push
    echo "Creating new git tag $TAG"
    git tag "$TAG" -m "$TAG"
    git push --tags
else
    echo "Cancelled"
    exit 1
fi
