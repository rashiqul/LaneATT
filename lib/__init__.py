import sys
import os

# Add NMS module to Python path
nms_path = os.path.join(os.path.dirname(__file__), 'nms', 'src')
if nms_path not in sys.path:
    sys.path.insert(0, nms_path)
