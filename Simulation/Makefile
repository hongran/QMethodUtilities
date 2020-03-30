# Grab the targets and sources as two batches
SOURCES = $(wildcard src/*.cxx)
HEADERS = $(wildcard include/*)
EXEC = bin/QSimulation

# Versioning info
MAJOR=0
MINOR=0.0
PREFIX=/home/hongran/QMethodUtilities/Simulation
SONAME=libqsim.so
#LIBNAME=$(SONAME).$(MAJOR).$(MINOR)
LIBNAME=$(SONAME)

# Figure out the architecture
UNAME_S = $(shell uname -s)

# Clang compiler
ifeq ($(UNAME_S), Darwin)
	CXX = clang++
	CC  = clang
	FLAGS = -std=c++11
	LDCONFIG = cp $(PREFIX)/lib/$(LIBNAME) $(PREFIX)/lib/$(SONAME).$(MAJOR)
endif

# Gnu compiler
ifeq ($(UNAME_S), Linux)
	CXX = g++
	CC  = gcc
	FLAGS = -std=c++11
	LDCONFIG = ldconfig -n -v $(PREFIX)/lib
endif

# Some optional compiler flags
ifdef DEBUG
	CC += -g -pg
	CXX += -g -pg
else
	CC += -O3
	CXX += -O3
endif

FLAGS += -Iinclude -I$(FFTW_INCLUDE_DIR)

LIBS += -lm -lfftw3 

ifdef FFTW_INCLUDE_DIR
FLAGS += -I$(FFTW_INCLUDE_DIR)
endif

ifdef FFTW_LIBRARY
LIBS += -L$(FFTW_LIBRARY)
endif

ROOT_FLAGS += $(shell root-config --cflags)
ROOT_LIBS  += $(shell root-config --libs)

#GPU_FLAGS := -std=c++11 -I$(CUDA_HOME)/include -Iinclude
GPU_FLAGS := -I$(CUDA_HOME)/include -Iinclude
GPU_FLAGS += --compiler-options '-Wall -fPIC'
GPU_FLAGS += -m64 
GPU_LIBS += -L$(CUDA_HOME)/lib64 -lcudart -lcufft -lcusolver -lcublas -lcurand -lm -lgomp

EXEC_FLAGS := -I$(CUDA_HOME)/include -Iinclude -I$(FFTW_INCLUDE_DIR)
EXEC_LIBS := -L./lib -lqsim

NVCC := nvcc -ccbin g++

GENCODE := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70

all: lib/$(LIBNAME) $(EXEC)

obj/QSim.o: src/QSim.cu $(HEADERS)
	$(NVCC) $(FLAGS) $(GPU_FLAGS) $(GENCODE) -o $@ -c $<

lib/$(LIBNAME): obj/QSim.o
	$(CXX) -shared -fPIC $+ -o $@ $(LIBS) $(GPU_LIBS)

obj/QSimulation.o: src/QSimulation.cpp $(HEADERS)
	$(CXX) $(EXEC_FLAGS) $(ROOT_FLAGS) -o $@ -c $<

$(EXEC): obj/QSimulation.o
	 $(CXX) -o $@ $^ $(ROOT_LIBS) $(EXEC_LIBS)
  
install:
	mkdir -p $(PREFIX)/lib
	cp lib/$(LIBNAME) $(PREFIX)/lib
	mkdir -p $(PREFIX)/include
	cp -r $(HEADERS) $(PREFIX)/include
	ln -sf $(PREFIX)/lib/$(LIBNAME) $(PREFIX)/lib/$(SONAME)
	$(LDCONFIG)


uninstall:
	rm -f $(PREFIX)/lib/$(SONAME)*
	rm -rf $(patsubst include/%,$(PREFIX)/include/%,$(HEADERS))
clean:
	rm -f obj/* lib/*
