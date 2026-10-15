# make            run all four demos on the synthetic source, refresh out/*.png
#
# Pass SOURCE=0 to run against a real camera instead:
#   make SOURCE=0
PY     ?= python
SOURCE ?= demo
DEMOS   = demo1_pixel_noise.py demo2_frame_averaging.py \
          demo3_tracking_kalman.py demo4_background_gaussian.py

.PHONY: all

all:
	@for d in $(DEMOS); do \
		echo "--- $$d"; \
		$(PY) $$d --source $(SOURCE) --headless || exit 1; \
	done
