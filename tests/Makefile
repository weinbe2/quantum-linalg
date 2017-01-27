.SUFFIXES:
.SUFFIXES: .o .c

#============================================================

C_SOURCES =  ../inverters/generic_cg.cpp ../inverters/generic_bicgstab.cpp ../inverters/generic_gcr.cpp
C_OBJS    = generic_bicgstab.o generic_cg.o generic_gcr.o
C_INCLUDES  = generic_vector.h verbosity.h
C_LIBS    = -lm
EXAMPLES = square_laplace

CCX = g++
#CXXFLAGS =   -std=c99 -g -O2  
CXXFLAGS = -O2 -Wall -std=c++11 -I../inverters/ -I../verbosity/ -I../blas/


#  = -DHAVE_CONFIG_H -I   -g -O2  -MT
# -g -O2  $(INC)

#============================================================

all: $(EXAMPLES)

%.o:%.cpp       $(INCLUDES)
	$(CCX) -c  $(CXXFLAGS) $(C_SOURCES)

square_laplace : square_laplace.o $(C_OBJS)
	$(CCX) $@.cpp $(CXXFLAGS)  -o $@ $(C_OBJS) $(C_LIBS)


#%.o:%.cpp	$(INCLUDES)
#	$(CCX) -c  $(CXXFLAGS) $(*:=.cpp)

#square_laplace :  $(C_OBJS)
#	$(CCX) $@.cpp $(CXXFLAGS)  -o $@ $(C_OBJS) $(C_LIBS)


#============================================================

ALL_SOURCES = Makefile $(C_SOURCES) $(INCLUDES) $(EXAMPLES)

clean: 
	rm -f $(TARGET) $(C_OBJS) $(EXAMPLES) $(C_EXAMPLES_OBJ)

TAGS:	$(ALL_SOURCES)
	etags $(ALL_SOURCES)

tar: $(ALL_SOURCES) 
	tar cvf $(TARGET).tar $(ALL_SOURCES) 

ps: $(ALL SOURCES)
	enscript -pcode.ps $(ALL_SOURCES)


