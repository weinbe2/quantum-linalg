.SUFFIXES:
.SUFFIXES: .o .c

#============================================================

C_SOURCES = 
C_OBJS    = 
C_INCLUDES  = 
C_LIBS    = -lm
EXAMPLES = batch_qr
C_EXAMPLES_OBJ = batch_qr.o

CCX = g++
#CXXFLAGS =   -std=c99 -g -O2  
CXXFLAGS = -O2 -g -Wall -std=c++11 -I../../


#  = -DHAVE_CONFIG_H -I   -g -O2  -MT
# -g -O2  $(INC)

#============================================================

all: $(EXAMPLES)

# There are no dependent files!
#%.o:%.cpp       $(INCLUDES)
#	$(CCX) -c  $(CXXFLAGS) $(C_SOURCES)
#
#square_laplace : square_laplace.o $(C_OBJS)
#	$(CCX) $@.cpp $(CXXFLAGS)  -o $@ $(C_OBJS) $(C_LIBS)


%.o:%.cpp	$(INCLUDES)
	$(CCX) -c  $(CXXFLAGS) $(*:=.cpp)

batch_qr :  $(C_OBJS)
	$(CCX) $@.cpp $(CXXFLAGS)  -o $@ $(C_OBJS) $(C_LIBS)


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


