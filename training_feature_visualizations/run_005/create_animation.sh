#!/bin/bash
# Create animation from training visualizations

# Using imagemagick (install: sudo apt-get install imagemagick)
convert -delay 30 -loop 0 epoch_*.png training_animation.gif

# Or using ffmpeg (install: sudo apt-get install ffmpeg)
# ffmpeg -framerate 2 -pattern_type glob -i 'epoch_*.png' \
#        -c:v libx264 -pix_fmt yuv420p training_evolution.mp4
