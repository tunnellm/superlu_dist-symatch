CC = gcc
DFLAGS = 
AR = ar
RANLIB= ranlib
INCS = -Iinc $(OPTFLAGS)
# LIBS = -ldl -lnuma $(OPTLIBS)
LIBS = -ldl $(OPTLIBS)

#CFLAGS = -Wall -Wno-unused-variable -Wno-unused-label -Werror -fno-exceptions				\
-fstack-protector -Wl,-z,relro -Wl,-z,now -Wformat-security -Wpointer-arith	\
-Winit-self -Wl,-O1 -Wl,--no-undefined
CFLAGS = -w -Wno-unused-variable -fopenmp

# SOURCES=$(wildcard *.c)
SOURCES=wframe3-exec.c wframe3-lib.c
OBJECTS=$(patsubst %.c,%.o,$(SOURCES))
EXOBJS=$(filter-out wframe3-lib.o, $(OBJECTS))
LIBOBJS=$(filter-out wframe3-exec.o, $(OBJECTS))
# LIBOBJS=$(OBJECTS)
EXECS=wframe3
LIBNAME=libsuitor

# If you want a library, build it into ``build'' directory


all: $(EXECS) lib


# Optimized build
opt: CFLAGS += -O3
opt: all

# Debugging build
dbg: CFLAGS += -g -ggdb3
dbg: all

# Profiling build
prof: CFLAGS += -pg
prof: all

# Define as more needed...


# Compilation
wframe3: build $(EXOBJS)
	$(CC) $(DFLAGS) $(CFLAGS) -o bin/$@ $(EXOBJS) $(INCS) $(LIBS)

%.o: %.c
	$(CC) $(DFLAGS) $(CFLAGS) -c $< -o $@ $(INCS)


build:
	test -d lib || mkdir lib
	test -d bin || mkdir bin


# Library (static)
lib: build $(LIBOBJS)
	$(AR) r lib/$(LIBNAME).a $(LIBOBJS)
	$(RANLIB) lib/$(LIBNAME).a
	rm -f $(LIBOBJS)



.PHONY: clean
clean:
	rm -f $(OBJECTS)
