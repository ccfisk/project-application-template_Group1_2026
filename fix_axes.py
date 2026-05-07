"""
Run this to fix the _make_axes_mock function in test_features.py
"""

with open('test_features.py', 'r') as f:
    content = f.read()

# Find and replace the entire _make_axes_mock function
# regardless of exact whitespace differences

import re

old_pattern = r'def _make_axes_mock\(\):.*?return fig, axes_array'
new_func = '''def _make_axes_mock():
    """
    feature3 does axes.reshape(3, 2) then axes[row][col].
    We need a real numpy object array of shape (3,2).
    """
    axes_array = np.empty((3, 2), dtype=object)
    for i in range(3):
        for j in range(2):
            axes_array[i, j] = MagicMock()
    fig = MagicMock()
    return fig, axes_array'''

new_content = re.sub(old_pattern, new_func, content, flags=re.DOTALL)

if new_content != content:
    with open('test_features.py', 'w') as f:
        f.write(new_content)
    print("Fix 3 applied successfully.")
else:
    print("Could not find the function to replace.")

print("Now run: python -m pytest test_features.py -v")
