"""
Run this once to fix test_features.py automatically.
Just double-click it or run: python fix_tests.py
"""

import re

with open('test_features.py', 'r') as f:
    content = f.read()

# ---------------------------------------------------------------
# Fix 1: Feature 3 axes mock — use proper numpy object array
# ---------------------------------------------------------------
old3 = """    ax = MagicMock()
    axes_array = np.array([[ax, ax], [ax, ax], [ax, ax]])
    fig = MagicMock()
    return fig, axes_array"""

new3 = """    axes_array = np.empty((3, 2), dtype=object)
    for i in range(3):
        for j in range(2):
            axes_array[i, j] = MagicMock()
    fig = MagicMock()
    return fig, axes_array"""

if old3 in content:
    content = content.replace(old3, new3)
    print("Fix 3 applied.")
else:
    print("Fix 3 not needed or already applied.")

# ---------------------------------------------------------------
# Fix 2: Feature 1 _run_feature1 — also patch Series.plot so
# pandas doesn't try to call real matplotlib internals
# ---------------------------------------------------------------
old1 = """    def _run_feature1(self, issues):
        \"\"\"Helper: run feature1 with a fixed naive 'now' and given issues.\"\"\"
        fixed_now = pd.Timestamp('2026-01-01')  # tz-naive
        with patch('feature1_analysis.DataLoader') as mock_loader, \\
             patch('feature1_analysis.plt'), \\
             patch.object(pd.Timestamp, 'now', return_value=fixed_now):
            from feature1_analysis import analysis_time_commit_hist
            mock_loader.return_value.get_issues.return_value = issues
            analysis_time_commit_hist().run()"""

new1 = """    def _run_feature1(self, issues):
        \"\"\"Helper: run feature1 with a fixed naive 'now' and given issues.\"\"\"
        fixed_now = pd.Timestamp('2026-01-01')  # tz-naive
        mock_series = MagicMock()
        with patch('feature1_analysis.DataLoader') as mock_loader, \\
             patch('feature1_analysis.plt'), \\
             patch.object(pd.Timestamp, 'now', return_value=fixed_now), \\
             patch.object(pd.Series, 'plot', return_value=mock_series):
            from feature1_analysis import analysis_time_commit_hist
            mock_loader.return_value.get_issues.return_value = issues
            analysis_time_commit_hist().run()"""

if old1 in content:
    content = content.replace(old1, new1)
    print("Fix 1 applied.")
else:
    print("Fix 1 not needed or already applied.")

# ---------------------------------------------------------------
# Also fix the separate test_run_calls_plt_show which has its
# own inline patch block (not using _run_feature1)
# ---------------------------------------------------------------
old1b = """    def test_run_calls_plt_show(self):
        fixed_now = pd.Timestamp('2026-01-01')
        with patch('feature1_analysis.DataLoader') as mock_loader, \\
             patch('feature1_analysis.plt') as mock_plt, \\
             patch.object(pd.Timestamp, 'now', return_value=fixed_now):
            from feature1_analysis import analysis_time_commit_hist
            mock_loader.return_value.get_issues.return_value = [
                FakeIssue(created_date=datetime(2025, 11, 1)),
            ]
            analysis_time_commit_hist().run()
            mock_plt.show.assert_called_once()"""

new1b = """    def test_run_calls_plt_show(self):
        fixed_now = pd.Timestamp('2026-01-01')
        mock_series = MagicMock()
        with patch('feature1_analysis.DataLoader') as mock_loader, \\
             patch('feature1_analysis.plt') as mock_plt, \\
             patch.object(pd.Timestamp, 'now', return_value=fixed_now), \\
             patch.object(pd.Series, 'plot', return_value=mock_series):
            from feature1_analysis import analysis_time_commit_hist
            mock_loader.return_value.get_issues.return_value = [
                FakeIssue(created_date=datetime(2025, 11, 1)),
            ]
            analysis_time_commit_hist().run()
            mock_plt.show.assert_called_once()"""

if old1b in content:
    content = content.replace(old1b, new1b)
    print("Fix 1b applied.")
else:
    print("Fix 1b not needed or already applied.")

with open('test_features.py', 'w') as f:
    f.write(content)

print("\nAll done. Now run: python -m pytest test_features.py -v")
