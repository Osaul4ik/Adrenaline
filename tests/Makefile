# Host unit tests for the native resolution GE list patcher.
#   make -C tests        build and run (AddressSanitizer + UBSan)

CC      ?= gcc
SRC     := ../cef/systemctrl
CFLAGS  := -std=gnu11 -O1 -g -Wall -Wextra -Werror -Wno-unused-parameter \
           -fsanitize=address,undefined -fno-sanitize-recover=undefined \
           -DGE_NATIVE_HOST_TEST -I$(SRC)
LDLIBS  := -lm

all: run

ge_native_test: ge_native_test.c $(SRC)/ge_native_list.c $(SRC)/ge_native.h
	$(CC) $(CFLAGS) -o $@ ge_native_test.c $(SRC)/ge_native_list.c $(LDLIBS)

# same tests with the frame trace compiled in (debug builds)
ge_native_test_trace: ge_native_test.c $(SRC)/ge_native_list.c $(SRC)/ge_native.h
	$(CC) $(CFLAGS) -DGE_NATIVE_TRACE -o $@ ge_native_test.c $(SRC)/ge_native_list.c $(LDLIBS)

run: ge_native_test ge_native_test_trace
	./ge_native_test
	./ge_native_test_trace

clean:
	rm -f ge_native_test ge_native_test_trace

.PHONY: all run clean
