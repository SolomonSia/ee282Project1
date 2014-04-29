#####################################################################
# EE282 Programming Assignment 1:
# Optimization of Matrix Multiplication
#
# Updated by: mgao12	04/14/2014
#####################################################################

## Provide your own options or compiler here.
CC = gcc
#CC = icc
CFLAGS += -O3 -g -Wall -ftree-vectorize -funroll-loops -Wl,-no-as-needed
## Add pthread if you use multi-threaded
CFLAGS += -pthread


##########################################################
## You shouldn't need to edit anything past this point. ##
##########################################################

APP = matmul
SRCS = driver.c matmul.c utils.c

BLAS=/usr/local/atlas/
ifneq ($(shell ls $(BLAS)/lib/libcblas.a 2>/dev/null),)
  LDFLAGS += -L$(BLAS)/lib/ -lcblas -latlas
  CFLAGS += -DBLAS -I$(BLAS)/include/
endif

all: $(APP)

$(APP): $(SRCS:.c=.o)
	$(LINK.c) $^ -o $@ $(LDFLAGS)

%.o: %.c utils.h Makefile
	$(COMPILE.c) $< -o $@

clean:
	rm -f *.o *~ $(APP)
