#!/bin/bash
let a=0; 

for i in *.jpg; 
	do 
		let a=a+1; 
		b=`printf Bild_%03d.jpg $a`; 
		#echo "mv $i $b"; 
		mv $i $b; 
done
