CC = gcc
LIBS = -lfftw3 -lm -lpthread
CFLAGS = -O6 -ggdb -Wall -Wextra

HEAD = arena.h
OBJ = 

all: main

main: $(OBJ) $(HEAD)
	$(CC) $(CFLAGS) main.c $(OBJ) -o main $(LIBS)

run:
	./build/main

clean:
	rm -f main 
