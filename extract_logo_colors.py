from PIL import Image
from collections import Counter

# Load the logo image
img = Image.open('static/img/logo.png').convert('RGBA')

# Resize for faster processing
img = img.resize((100, 100))

# Get all pixels, filter out transparent/white
pixels = [pixel for pixel in img.getdata() if pixel[3] > 0 and not (pixel[0] > 240 and pixel[1] > 240 and pixel[2] > 240)]

# Count most common colors
color_counts = Counter(pixels)
most_common = color_counts.most_common(6)

print('Most common colors (RGBA):')
for color, count in most_common:
    print(f'{color} (count: {count})')

# Convert to HEX
def rgba_to_hex(rgba):
    return '#{:02X}{:02X}{:02X}'.format(rgba[0], rgba[1], rgba[2])

print('\nMost common colors (HEX):')
for color, count in most_common:
    print(f'{rgba_to_hex(color)} (count: {count})') 