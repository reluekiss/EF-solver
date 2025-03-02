CC = gcc
LIBS = -lfftw3 -lm -lpthread
CFLAGS = -O6 -ggdb -Wall -Wextra

OBJ = 

all: main

main: $(OBJ)
	$(CC) $(CFLAGS) main.c $(OBJ) -o main $(LIBS)

run:
	./build/main

clean:
	rm -f main 
