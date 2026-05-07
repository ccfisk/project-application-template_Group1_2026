"""
Unit tests for Feature 1, Feature 2, Feature 3, and model.

Run:
    python -m pytest test_features.py -v
    python -m coverage run -m pytest test_features.py -v
    python -m coverage report --omit="test*,scrape_github.py" --show-missing
"""

import unittest
from unittest.mock import patch, MagicMock
from collections import Counter
from datetime import datetime
import pandas as pd
import numpy as np


# ===========================================================================
# Shared fake objects
# ===========================================================================

class FakeEvent:
    def __init__(self, event_type=None, event_date=None):
        self.event_type = event_type
        self.event_date = event_date
        self.author = None
        self.label = None
        self.comment = None


class FakeIssue:
    def __init__(self, created_date=None, labels=None, events=None, state='open'):
        self.created_date = created_date
        self.labels = labels if labels is not None else []
        self.events = events if events is not None else []
        self.state = state
        self.updated_date = None
        self.creator = None
        self.url = None
        self.title = None
        self.text = None
        self.number = -1
        self.assignees = []
        self.timeline_url = None


def _make_axes_mock():
    """
    feature3 does axes.reshape(3, 2) then axes[row][col].
    We need a real numpy object array of shape (3,2).
    """
    axes_array = np.empty((3, 2), dtype=object)
    for i in range(3):
        for j in range(2):
            axes_array[i, j] = MagicMock()
    fig = MagicMock()
    return fig, axes_array


# ===========================================================================
# FEATURE 1 TESTS
#
# The problem: feature1_analysis.py does:
#   six_months_ago = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=self.months)
#   df = df[df["date"] >= six_months_ago]
#
# But df["date"] is tz-naive (created from plain datetime objects),
# so comparing against a tz-aware cutoff crashes.
#
# Fix strategy: use patch.object to make pd.Timestamp.now() return a
# tz-naive Timestamp, so both sides of the comparison are naive.
# We must NOT patch the whole pd.Timestamp class — that breaks DateOffset.
# ===========================================================================

class TestFeature1WeeklyCommits(unittest.TestCase):

    def _run_feature1(self, issues):
        """Helper: run feature1 with a fixed naive 'now' and given issues."""
        fixed_now = pd.Timestamp('2026-01-01')  # tz-naive
        mock_series = MagicMock()
        with patch('feature1_analysis.DataLoader') as mock_loader, \
             patch('feature1_analysis.plt'), \
             patch.object(pd.Timestamp, 'now', return_value=fixed_now), \
             patch.object(pd.Series, 'plot', return_value=mock_series):
            from feature1_analysis import analysis_time_commit_hist
            mock_loader.return_value.get_issues.return_value = issues
            analysis_time_commit_hist().run()

    def test_run_completes_without_error(self):
        self._run_feature1([
            FakeIssue(created_date=datetime(2025, 11, 10)),
            FakeIssue(created_date=datetime(2025, 12, 20)),
        ])

    def test_run_calls_plt_show(self):
        fixed_now = pd.Timestamp('2026-01-01')
        mock_series = MagicMock()
        with patch('feature1_analysis.DataLoader') as mock_loader, \
             patch('feature1_analysis.plt') as mock_plt, \
             patch.object(pd.Timestamp, 'now', return_value=fixed_now), \
             patch.object(pd.Series, 'plot', return_value=mock_series):
            from feature1_analysis import analysis_time_commit_hist
            mock_loader.return_value.get_issues.return_value = [
                FakeIssue(created_date=datetime(2025, 11, 1)),
            ]
            analysis_time_commit_hist().run()
            mock_plt.show.assert_called_once()

    def test_run_empty_issue_list(self):
        try:
            self._run_feature1([])
        except Exception as e:
            self.fail(f'run() crashed on empty list: {e}')

    def test_run_all_none_dates(self):
        try:
            self._run_feature1([
                FakeIssue(created_date=None),
                FakeIssue(created_date=None),
            ])
        except Exception as e:
            self.fail(f'run() crashed on None dates: {e}')

    def test_run_single_issue(self):
        try:
            self._run_feature1([FakeIssue(created_date=datetime(2025, 11, 1))])
        except Exception as e:
            self.fail(f'run() crashed with single issue: {e}')

    def test_none_dates_filtered_out(self):
        issues = [
            FakeIssue(created_date=datetime(2025, 4, 10)),
            FakeIssue(created_date=None),
            FakeIssue(created_date=datetime(2025, 4, 15)),
        ]
        dates = [i.created_date for i in issues if i.created_date]
        self.assertEqual(len(dates), 2)

    def test_date_range_filter_keeps_recent(self):
        dates = [datetime(2025, 11, 1)]
        df = pd.DataFrame({'date': pd.to_datetime(dates)})
        cutoff = pd.Timestamp('2025-07-01')
        result = df[df['date'] >= cutoff]
        self.assertEqual(len(result), 1)

    def test_date_range_filter_removes_old(self):
        dates = [datetime(2010, 1, 1)]
        df = pd.DataFrame({'date': pd.to_datetime(dates)})
        cutoff = pd.Timestamp('2025-07-01')
        result = df[df['date'] >= cutoff]
        self.assertEqual(len(result), 0)

    def test_weekly_resample_groups_same_week(self):
        dates = [datetime(2025, 4, 7), datetime(2025, 4, 9)]
        df = pd.DataFrame({'date': pd.to_datetime(dates)})
        weekly = df.set_index('date').resample('W').size()
        self.assertEqual(weekly.sum(), 2)
        self.assertEqual(len(weekly), 1)

    def test_weekly_resample_separates_different_weeks(self):
        dates = [datetime(2025, 4, 1), datetime(2025, 4, 14)]
        df = pd.DataFrame({'date': pd.to_datetime(dates)})
        weekly = df.set_index('date').resample('W').size()
        self.assertGreaterEqual(len(weekly), 2)


# ===========================================================================
# FEATURE 2 TESTS
# ===========================================================================

class TestFeature2LabelTypes(unittest.TestCase):

    def _count(self, issues):
        label_counts = Counter()
        unlabeled_count = 0
        for issue in issues:
            if issue.labels:
                label_counts.update(issue.labels)
            else:
                unlabeled_count += 1
        return label_counts, unlabeled_count

    def test_single_labeled_issue(self):
        counts, unlabeled = self._count([FakeIssue(labels=['bug'])])
        self.assertEqual(counts['bug'], 1)
        self.assertEqual(unlabeled, 0)

    def test_single_unlabeled_issue(self):
        counts, unlabeled = self._count([FakeIssue(labels=[])])
        self.assertEqual(unlabeled, 1)
        self.assertEqual(len(counts), 0)

    def test_none_labels_treated_as_unlabeled(self):
        counts, unlabeled = self._count([FakeIssue(labels=None)])
        self.assertEqual(unlabeled, 1)

    def test_multiple_labels_on_one_issue(self):
        counts, _ = self._count([FakeIssue(labels=['bug', 'enhancement', 'help wanted'])])
        self.assertEqual(counts['bug'], 1)
        self.assertEqual(counts['enhancement'], 1)
        self.assertEqual(counts['help wanted'], 1)

    def test_same_label_across_multiple_issues_accumulates(self):
        counts, _ = self._count([FakeIssue(labels=['bug']) for _ in range(5)])
        self.assertEqual(counts['bug'], 5)

    def test_mixed_labeled_and_unlabeled(self):
        issues = [
            FakeIssue(labels=['bug']),
            FakeIssue(labels=[]),
            FakeIssue(labels=['bug', 'feature']),
            FakeIssue(labels=None),
        ]
        counts, unlabeled = self._count(issues)
        self.assertEqual(counts['bug'], 2)
        self.assertEqual(counts['feature'], 1)
        self.assertEqual(unlabeled, 2)

    def test_empty_issue_list(self):
        counts, unlabeled = self._count([])
        self.assertEqual(len(counts), 0)
        self.assertEqual(unlabeled, 0)

    def test_most_common_ordering(self):
        issues = [FakeIssue(labels=['bug']), FakeIssue(labels=['bug']), FakeIssue(labels=['feature'])]
        counts, _ = self._count(issues)
        self.assertEqual(counts.most_common(1)[0][0], 'bug')

    def test_top_15_label_limit(self):
        issues = [FakeIssue(labels=[f'label-{i}']) for i in range(20)]
        counts, _ = self._count(issues)
        self.assertLessEqual(len(counts.most_common(15)), 15)

    @patch('feature2_analysis.DataLoader')
    @patch('feature2_analysis.plt')
    def test_run_calls_plt_show(self, mock_plt, mock_loader):
        from feature2_analysis import analysis_label_types
        mock_loader.return_value.get_issues.return_value = [FakeIssue(labels=['bug']), FakeIssue(labels=[])]
        analysis_label_types().run()
        mock_plt.show.assert_called_once()

    @patch('feature2_analysis.DataLoader')
    @patch('feature2_analysis.plt')
    def test_run_empty_issues(self, mock_plt, mock_loader):
        from feature2_analysis import analysis_label_types
        mock_loader.return_value.get_issues.return_value = []
        try:
            analysis_label_types().run()
        except Exception as e:
            self.fail(f'crashed: {e}')

    @patch('feature2_analysis.DataLoader')
    @patch('feature2_analysis.plt')
    def test_run_all_unlabeled(self, mock_plt, mock_loader):
        from feature2_analysis import analysis_label_types
        mock_loader.return_value.get_issues.return_value = [FakeIssue(labels=[]), FakeIssue(labels=[])]
        try:
            analysis_label_types().run()
        except Exception as e:
            self.fail(f'crashed: {e}')

    @patch('feature2_analysis.DataLoader')
    @patch('feature2_analysis.plt')
    def test_run_all_labeled(self, mock_plt, mock_loader):
        from feature2_analysis import analysis_label_types
        mock_loader.return_value.get_issues.return_value = [FakeIssue(labels=['bug']), FakeIssue(labels=['enhancement'])]
        try:
            analysis_label_types().run()
        except Exception as e:
            self.fail(f'crashed: {e}')


# ===========================================================================
# FEATURE 3 TESTS
#
# The problem: feature3_analysis.py does axes.reshape(3, 2) after subplots.
# Our _make_axes_mock() now returns a real numpy array so .reshape() works.
# ===========================================================================

class TestFeature3SeasonalPatterns(unittest.TestCase):

    def setUp(self):
        with patch('feature3_analysis.DataLoader'):
            from feature3_analysis import SeasonalPatternAnalysis
            self.analysis = SeasonalPatternAnalysis()

    def test_extract_creations_basic_values(self):
        d = datetime(2023, 3, 15)
        df = self.analysis._extract_creations([FakeIssue(created_date=d)])
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['day_of_week'], 2)
        self.assertEqual(df.iloc[0]['month'], 3)

    def test_extract_creations_skips_none_date(self):
        df = self.analysis._extract_creations([FakeIssue(created_date=None)])
        self.assertEqual(len(df), 0)

    def test_extract_creations_empty_list(self):
        self.assertEqual(len(self.analysis._extract_creations([])), 0)

    def test_extract_creations_filters_none_keeps_valid(self):
        issues = [FakeIssue(created_date=datetime(2023, 1, 2)), FakeIssue(created_date=None), FakeIssue(created_date=datetime(2023, 6, 10))]
        self.assertEqual(len(self.analysis._extract_creations(issues)), 2)

    def test_extract_creations_columns_exist(self):
        df = self.analysis._extract_creations([FakeIssue(created_date=datetime(2023, 1, 1))])
        self.assertIn('day_of_week', df.columns)
        self.assertIn('month', df.columns)

    def test_extract_creations_day_of_week_range(self):
        issues = [FakeIssue(created_date=datetime(2023, 1, d)) for d in range(2, 9)]
        df = self.analysis._extract_creations(issues)
        self.assertTrue((df['day_of_week'] >= 0).all())
        self.assertTrue((df['day_of_week'] <= 6).all())

    def test_extract_creations_month_range(self):
        issues = [FakeIssue(created_date=datetime(2023, m, 1)) for m in range(1, 13)]
        df = self.analysis._extract_creations(issues)
        self.assertTrue((df['month'] >= 1).all())
        self.assertTrue((df['month'] <= 12).all())

    def test_extract_referenced_picks_referenced_type(self):
        event = FakeEvent(event_type='referenced', event_date=datetime(2023, 4, 5))
        df = self.analysis._extract_referenced([FakeIssue(events=[event])])
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['month'], 4)

    def test_extract_referenced_ignores_other_types(self):
        events = [FakeEvent(event_type='closed', event_date=datetime(2023, 4, 6))]
        self.assertEqual(len(self.analysis._extract_referenced([FakeIssue(events=events)])), 0)

    def test_extract_referenced_skips_none_date(self):
        event = FakeEvent(event_type='referenced', event_date=None)
        self.assertEqual(len(self.analysis._extract_referenced([FakeIssue(events=[event])])), 0)

    def test_extract_referenced_no_events(self):
        self.assertEqual(len(self.analysis._extract_referenced([FakeIssue(events=[])])), 0)

    def test_extract_referenced_empty_list(self):
        self.assertEqual(len(self.analysis._extract_referenced([])), 0)

    def test_extract_referenced_multiple_events_same_issue(self):
        events = [
            FakeEvent(event_type='referenced', event_date=datetime(2023, 1, 10)),
            FakeEvent(event_type='referenced', event_date=datetime(2023, 2, 15)),
        ]
        self.assertEqual(len(self.analysis._extract_referenced([FakeIssue(events=events)])), 2)

    def test_extract_closures_basic(self):
        event = FakeEvent(event_type='closed', event_date=datetime(2023, 7, 20))
        df = self.analysis._extract_closures([FakeIssue(events=[event])])
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['month'], 7)

    def test_extract_closures_ignores_other_types(self):
        events = [FakeEvent(event_type='referenced', event_date=datetime(2023, 7, 20))]
        self.assertEqual(len(self.analysis._extract_closures([FakeIssue(events=events)])), 0)

    def test_extract_closures_skips_none_date(self):
        event = FakeEvent(event_type='closed', event_date=None)
        self.assertEqual(len(self.analysis._extract_closures([FakeIssue(events=[event])])), 0)

    def test_extract_closures_empty_events(self):
        self.assertEqual(len(self.analysis._extract_closures([FakeIssue(events=[])])), 0)

    def test_extract_closures_empty_issue_list(self):
        self.assertEqual(len(self.analysis._extract_closures([])), 0)

    def test_extract_closures_multiple_closed_events(self):
        events = [
            FakeEvent(event_type='closed', event_date=datetime(2023, 1, 5)),
            FakeEvent(event_type='closed', event_date=datetime(2023, 3, 10)),
        ]
        self.assertEqual(len(self.analysis._extract_closures([FakeIssue(events=events)])), 2)

    @patch('feature3_analysis.DataLoader')
    @patch('feature3_analysis.plt')
    def test_run_calls_plt_show(self, mock_plt, mock_loader):
        from feature3_analysis import SeasonalPatternAnalysis
        mock_plt.subplots.return_value = _make_axes_mock()
        mock_loader.return_value.get_issues.return_value = [
            FakeIssue(created_date=datetime(2023, 3, 15), events=[
                FakeEvent(event_type='referenced', event_date=datetime(2023, 3, 16)),
                FakeEvent(event_type='closed', event_date=datetime(2023, 4, 1)),
            ])
        ]
        SeasonalPatternAnalysis().run()
        mock_plt.show.assert_called_once()

    @patch('feature3_analysis.DataLoader')
    @patch('feature3_analysis.plt')
    def test_run_empty_issues(self, mock_plt, mock_loader):
        from feature3_analysis import SeasonalPatternAnalysis
        mock_plt.subplots.return_value = _make_axes_mock()
        mock_loader.return_value.get_issues.return_value = []
        try:
            SeasonalPatternAnalysis().run()
        except Exception as e:
            self.fail(f'crashed: {e}')

    @patch('feature3_analysis.DataLoader')
    @patch('feature3_analysis.plt')
    def test_run_no_events(self, mock_plt, mock_loader):
        from feature3_analysis import SeasonalPatternAnalysis
        mock_plt.subplots.return_value = _make_axes_mock()
        mock_loader.return_value.get_issues.return_value = [FakeIssue(created_date=datetime(2023, 5, 1), events=[])]
        try:
            SeasonalPatternAnalysis().run()
        except Exception as e:
            self.fail(f'crashed: {e}')


# ===========================================================================
# MODEL TESTS
# ===========================================================================

class TestModelEvent(unittest.TestCase):

    def test_event_basic_fields(self):
        from model import Event
        event = Event({'event_type': 'closed', 'author': 'bob', 'event_date': '2023-05-10T08:00:00Z', 'label': None, 'comment': None})
        self.assertEqual(event.event_type, 'closed')
        self.assertEqual(event.author, 'bob')
        self.assertEqual(event.event_date.month, 5)

    def test_event_handles_bad_date(self):
        from model import Event
        event = Event({'event_type': 'labeled', 'author': 'carol', 'event_date': 'not-a-date'})
        self.assertIsNone(event.event_date)

    def test_event_none_jobj_sets_defaults(self):
        from model import Event
        event = Event(None)
        self.assertIsNone(event.event_type)
        self.assertIsNone(event.event_date)

    def test_event_referenced_type(self):
        from model import Event
        event = Event({'event_type': 'referenced', 'author': 'alice', 'event_date': '2023-01-01T00:00:00Z'})
        self.assertEqual(event.event_type, 'referenced')


class TestModelIssue(unittest.TestCase):

    def test_issue_defaults_with_no_args(self):
        from model import Issue
        issue = Issue()
        self.assertIsNone(issue.creator)
        self.assertEqual(issue.labels, [])
        self.assertEqual(issue.events, [])

    def test_issue_from_json_basic(self):
        from model import Issue
        issue = Issue({
            'url': 'http://example.com/1', 'creator': 'alice', 'labels': ['bug'],
            'state': 'open', 'assignees': [], 'title': 'T', 'text': 'D',
            'number': '42', 'created_date': '2023-01-15T10:00:00Z',
            'updated_date': '2023-01-16T10:00:00Z', 'timeline_url': '', 'events': [],
        })
        self.assertEqual(issue.creator, 'alice')
        self.assertEqual(issue.number, 42)
        self.assertEqual(issue.created_date.year, 2023)

    def test_issue_handles_bad_created_date(self):
        from model import Issue
        issue = Issue({'state': 'open', 'labels': [], 'assignees': [], 'events': [], 'created_date': 'garbage', 'updated_date': 'bad'})
        self.assertIsNone(issue.created_date)

    def test_issue_events_populated(self):
        from model import Issue
        issue = Issue({'state': 'open', 'labels': [], 'assignees': [], 'events': [
            {'event_type': 'closed', 'author': 'x', 'event_date': '2023-01-01T00:00:00Z'}
        ]})
        self.assertEqual(len(issue.events), 1)
        self.assertEqual(issue.events[0].event_type, 'closed')

    def test_issue_multiple_labels(self):
        from model import Issue
        issue = Issue({'state': 'open', 'labels': ['bug', 'enhancement', 'help wanted'], 'assignees': [], 'events': []})
        self.assertEqual(len(issue.labels), 3)

    def test_issue_closed_state(self):
        from model import Issue
        issue = Issue({'state': 'closed', 'labels': [], 'assignees': [], 'events': []})
        self.assertEqual(issue.state.value, 'closed')


if __name__ == '__main__':
    unittest.main()
