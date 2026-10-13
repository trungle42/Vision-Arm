CFLAGS = -std=c99 -Wall -Wextra -O2
ifeq ($(shell uname),Darwin)
LIB = libservo.dylib
else
LIB = libservo.so
endif

test: test_servo
	./test_servo

test_servo: test_servo.c servo.c servo.h
	$(CC) $(CFLAGS) test_servo.c servo.c -o test_servo -lm

# shared library so Python can call the same C code
lib: servo.c servo.h
	$(CC) $(CFLAGS) -shared -fPIC servo.c -o $(LIB)

clean:
	rm -f test_servo libservo.dylib libservo.so
