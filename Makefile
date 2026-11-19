BUILD_DIR := build
PYTHON ?= python3
GPU_BACKEND ?= auto
CMAKE_FLAGS := -G Ninja -DCMAKE_BUILD_TYPE=Release

.PHONY: all config build qrun clean

all: build

config:
	@if [ ! -f $(BUILD_DIR)/build.ninja ]; then \
	    GPU_CMAKE_FLAGS=$$($(PYTHON) scripts/detect_gpu_flags.py --backend "$(GPU_BACKEND)") || exit 1; \
	    cmake -B $(BUILD_DIR) $(CMAKE_FLAGS) $$GPU_CMAKE_FLAGS; \
	fi

build: config
	ninja -C $(BUILD_DIR)

qrun: build
	./$(BUILD_DIR)/app

clean:
	rm -rf $(BUILD_DIR)
