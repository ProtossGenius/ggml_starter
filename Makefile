BUILD_DIR := build
CMAKE_FLAGS := -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON

.PHONY: all config build qrun clean

all: build

config:
	@if [ ! -f $(BUILD_DIR)/build.ninja ]; then \
	    cmake -B $(BUILD_DIR) $(CMAKE_FLAGS); \
	fi

build: config
	ninja -C $(BUILD_DIR)

qrun: build
	./$(BUILD_DIR)/app

clean:
	rm -rf $(BUILD_DIR)
