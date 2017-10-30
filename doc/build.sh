FORMAT=snickery_overview  # wrapper 

rm -f $FORMAT.blg $FORMAT.log $FORMAT.aux $FORMAT.cb $FORMAT.pdf $FORMAT.bbl $FORMAT.cb2

pdflatex $FORMAT.tex
bibtex $FORMAT
pdflatex $FORMAT.tex
pdflatex $FORMAT.tex

rm -f $FORMAT.blg $FORMAT.log $FORMAT.aux $FORMAT.cb              $FORMAT.bbl $FORMAT.cb2

