# Host unit tests for the native resolution GE list patcher and the SGX
# renderer's GE list walker.
#   make -C tests        build and run (AddressSanitizer + UBSan)

CC      ?= gcc
SRC     := ../cef/systemctrl
CFLAGS  := -std=gnu11 -O1 -g -Wall -Wextra -Werror -Wno-unused-parameter \
           -fsanitize=address,undefined -fno-sanitize-recover=undefined \
           -DGE_NATIVE_HOST_TEST -I$(SRC)
LDLIBS  := -lm
USER    := ../user

all: run

ge_native_test: ge_native_test.c $(SRC)/ge_native_list.c $(SRC)/ge_native.h
	$(CC) $(CFLAGS) -o $@ ge_native_test.c $(SRC)/ge_native_list.c $(LDLIBS)

# same tests with the frame trace compiled in (debug builds)
ge_native_test_trace: ge_native_test.c $(SRC)/ge_native_list.c $(SRC)/ge_native.h
	$(CC) $(CFLAGS) -DGE_NATIVE_TRACE -o $@ ge_native_test.c $(SRC)/ge_native_list.c $(LDLIBS)

SGX_SRC := $(USER)/ge_sgx_walk.c $(USER)/ge_sgx_build.c $(USER)/ge_sgx_texconv.c
SGX_HDR := $(USER)/ge_sgx_walk.h $(USER)/ge_sgx_frame.h $(USER)/ge_sgx_texconv.h

ge_sgx_test: ge_sgx_test.c $(SGX_SRC) $(SGX_HDR)
	$(CC) $(filter-out -DGE_NATIVE_HOST_TEST -I$(SRC),$(CFLAGS)) -I$(USER) -o $@ ge_sgx_test.c $(SGX_SRC) $(LDLIBS)

run: ge_native_test ge_native_test_trace ge_sgx_test
	./ge_native_test
	./ge_native_test_trace
	./ge_sgx_test

clean:
	rm -f ge_native_test ge_native_test_trace ge_sgx_test

.PHONY: all run clean
