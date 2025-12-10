# CXX = g++
# CXXFLAGS = -std=c++17 -pthread -Wall -O2
# TARGET = test
# SRCS = test.cpp junctiond.cpp
# OBJS = $(SRCS:.cpp=.o)

# all: $(TARGET)

# $(TARGET): $(OBJS)
# 	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

# %.o: %.cpp
# 	$(CXX) $(CXXFLAGS) -c $< -o $@

# clean:
# 	rm -f $(OBJS) $(TARGET)
CXX = g++
CXXFLAGS = -std=c++17 -pthread -Wall -O2

# 1. Common Files (The Logic)
# These are used by BOTH the test and the server.
COMMON_SRCS = junctiond.cpp
COMMON_OBJS = $(COMMON_SRCS:.cpp=.o)

# 2. Target: Test (test.cpp)
TEST_TARGET = test
TEST_SRCS = test.cpp
TEST_OBJS = $(TEST_SRCS:.cpp=.o)

# 3. Target: Server (main.cpp)
SERVER_TARGET = server
SERVER_SRCS = main.cpp
SERVER_OBJS = $(SERVER_SRCS:.cpp=.o)

# Default: Build both executables
all: $(TEST_TARGET) $(SERVER_TARGET)

# --- Build Rules ---

# Build the Test Binary
$(TEST_TARGET): $(TEST_OBJS) $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_OBJS) $(COMMON_OBJS)

# Build the Server Binary
$(SERVER_TARGET): $(SERVER_OBJS) $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(SERVER_OBJS) $(COMMON_OBJS)

# --- Helper Commands ---

# Convenience command to build AND run the server
run: $(SERVER_TARGET)
	@echo ">>> Starting Server..."
	./$(SERVER_TARGET)

# Compile .cpp files to .o files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TEST_TARGET) $(SERVER_TARGET)