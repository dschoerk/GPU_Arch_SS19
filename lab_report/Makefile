LATEX := pdflatex
BIBTEX := bibtex
RM := rm -f

SOURCES := \
	Makefile \
	main.tex \
	introduction.tex \
	implementations.tex \
	results.tex \
	refs.bib

GPU_Architectures_Lab_Report.pdf : main.pdf
	mv main.pdf GPU_Architectures_Lab_Report.pdf

main.pdf : main.tex $(SOURCES)
	$(LATEX) $<
	$(BIBTEX) $(basename $<)
	$(LATEX) $<
	$(LATEX) $<

all : GPU_Architectures_Lab_Report.pdf

.PHONY : all clean clobber

clean :
	$(RM) *.log *~ *.aux *.blg *.bbl *.toc *.out

clobber : clean
	$(RM) GPU_Architectures_Lab_Report.pdf

