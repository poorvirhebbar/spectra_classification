#!/bin/bash
# Create animation from training visualizations

# Try ffmpeg first (more commonly available)
if command -v ffmpeg &> /dev/null; then
    echo "Creating MP4 animation with ffmpeg..."
    ffmpeg -framerate 2 -pattern_type glob -i 'epoch_*.png' \
           -c:v libx264 -pix_fmt yuv420p training_evolution.mp4
    echo "✅ Animation saved as: training_evolution.mp4"
elif command -v convert &> /dev/null; then
    echo "Creating GIF animation with ImageMagick..."
    convert -delay 30 -loop 0 epoch_*.png training_animation.gif
    echo "✅ Animation saved as: training_animation.gif"
else
    echo "❌ Neither ffmpeg nor ImageMagick found!"
    echo "Install one of them:"
    echo "  - ffmpeg: brew install ffmpeg (macOS) or sudo apt-get install ffmpeg (Ubuntu)"
    echo "  - ImageMagick: brew install imagemagick (macOS) or sudo apt-get install imagemagick (Ubuntu)"
fi
