#####################################################################
# EE282 Programming Assignment 1:
# Optimization of Matrix Multiplication
#
# Created by: mgao12	04/14/2014
#####################################################################

#################################
## Don't change this Makefile! ##
#################################

APP_DIR = ./matmul
SIM_DIR = ./sim


all: compile

compile:
	$(MAKE) -C $(APP_DIR)

gensim:
	@./scripts/gen_scripts.py . ./scripts/nehalem.cfg.template

clean:
	$(MAKE) -C $(APP_DIR) clean --no-print-directory

cleansim:
	rm sim* -rf

cleanall: clean cleansim

.PHONY: clean cleansim cleanall

