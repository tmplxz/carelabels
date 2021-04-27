#!/usr/bin/env bash

# set -x
# set -v

mkdir -p imagenet
mkdir -p imagenet-{c,p}

# ImageNet-C

# Blur   : https://drive.google.com/file/d/15aiZpiQpQzYwWWSSpwKHf7wKo65j-oF4/view?usp=sharing
# Digital: https://drive.google.com/file/d/15vLMParMqQDpDe34qXTq1eAwZCK4OU_K/view?usp=sharing
# Extra  : https://drive.google.com/file/d/1LjYf2LMhSPfSdCYR9DFZj2N24ix84fds/view?usp=sharing
# Noise  : https://drive.google.com/file/d/1w05DJwhGz66zXTA0WK1ie9R54-ZmCtGB/view?usp=sharing
# Weather: https://drive.google.com/file/d/1IGdjgLrQocafIIYLs_r_skfOq24oNbB6/view?usp=sharing

cd imagemet-c

if [ ! -f blur.tar ]; then
  wget https://zenodo.org/record/2235448/files/blur.tar && tar -xf blur.tar
fi

if [ ! -f digital.tar ]; then
  wget https://zenodo.org/record/2235448/files/digital.tar && tar -xf digital.tar
fi

if [ ! -f extra.tar ]; then
  wget https://zenodo.org/record/2235448/files/extra.tar && tar -xf extra.tar
fi

if [ ! -f noise.tar ]; then
  wget https://zenodo.org/record/2235448/files/noise.tar && tar -xf noise.tar
fi

if [ ! -f weather.tar ]; then
  wget https://zenodo.org/record/2235448/files/weather.tar && tar -xf weather.tar
fi

cd ..
cd imagemet-p

if [ ! -f blur.tar ]; then
  wget https://zenodo.org/record/3565846/files/blur.tar && tar -xf blur.tar
fi

if [ ! -f digital.tar ]; then
  wget https://zenodo.org/record/3565846/files/digital.tar && tar -xf digital.tar
fi

if [ ! -f noise.tar ]; then
  wget https://zenodo.org/record/3565846/files/noise.tar && tar -xf noise.tar
fi

if [ ! -f weather.tar ]; then
  wget https://zenodo.org/record/3565846/files/weather.tar && tar -xf weather.tar
fi