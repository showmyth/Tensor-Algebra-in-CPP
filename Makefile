# Compiler and flags
CXX := g++
CXXFLAGS := -Wall -Wextra -Iinclude -std=c++23 -Werror

# Directories
SRC_DIR := src
INC_DIR := include
BIN_DIR := bin

# Source and object files
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(BIN_DIR)/%.o, $(SRCS))

# Target binary
TARGET := $(BIN_DIR)/main

# Default target
all: $(TARGET)

# Link object files into binary
$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Compile .cpp to .o
$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -rf $(BIN_DIR)/*

.PHONY: all clean

