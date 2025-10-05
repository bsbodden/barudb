#!/bin/bash
# Convert ASCII architecture diagram to PNG

cd /home/bsb/Code/hes/barudb/final_report

# Check if the text file exists
if [ ! -f "images/architecture.txt" ]; then
    echo "Error: architecture.txt not found in images directory"
    exit 1
fi

# Try to convert using figlet and imagemagick if they're available
if command -v figlet &> /dev/null && command -v convert &> /dev/null; then
    echo "Converting to PNG using ImageMagick..."
    convert -background white -fill black -font "Courier" -pointsize 12 \
        -border 20 -bordercolor white label:@images/architecture.txt \
        images/architecture.png
    echo "Converted to images/architecture.png"
    exit 0
fi

# Fallback to creating a simple HTML version
echo "Creating HTML version of the architecture diagram..."
echo '<!DOCTYPE html>
<html>
<head>
    <title>LSM Tree Architecture</title>
    <style>
        body { font-family: monospace; white-space: pre; }
    </style>
</head>
<body>' > images/architecture.html

cat images/architecture.txt >> images/architecture.html

echo '
</body>
</html>' >> images/architecture.html

echo "Created HTML version at images/architecture.html"
echo "Please open this file in a browser and take a screenshot to create architecture.png"