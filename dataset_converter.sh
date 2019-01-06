#!/bin/bash
mogrify -verbose -format jpg -background black -flatten -quality 100 *.PNG
