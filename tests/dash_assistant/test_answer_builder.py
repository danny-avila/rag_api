# tests/dash_assistant/test_answer_builder.py
"""Tests for answer builder functionality."""
import pytest
from typing import Dict, List, Any
from app.dash_assistant.serving.answer_builder import build_answer, AnswerBuilder


class TestAnswerBuilder:
    """Test answer builder functionality."""
    
    def test_build_answer_basic_structure(self):
        """Test basic structure of built answer."""
        query = "revenue dashboard"
        candidates = [
            {
                'entity_id': 1,
                'title': 'Revenue Analytics Dashboard',
                'url': 'https://superset.example.com/dashboard/1',
                'score': 0.95,
                'usage_score': 10.5,
                'signal_sources': ['vector', 'fts'],
                'matched_tokens': ['revenue', 'analytics'],
                'charts': [
                    {
                        'chart_id': 101,
                        'title': 'Monthly Revenue Trend',
                        'url': 'https://superset.example.com/chart/101',
                        'filters_default': {'time_range': 'last_30_days', 'region': 'US'}
                    }
                ]
            }
        ]
        
        result = build_answer(query, candidates)
        
        # Check basic structure
        assert isinstance(result, dict)
        assert 'results' in result
        assert 'suggested_filters' in result
        assert isinstance(result['results'], list)
        assert len(result['results']) == 1
        
        # Check result structure
        result_item = result['results'][0]
        assert 'title' in result_item
        assert 'url' in result_item
        assert 'score' in result_item
        assert 'charts' in result_item
        assert 'why' in result_item
        
        assert result_item['title'] == 'Revenue Analytics Dashboard'
        assert result_item['url'] == 'https://superset.example.com/dashboard/1'
        assert result_item['score'] == 0.95
        assert isinstance(result_item['charts'], list)
        assert len(result_item['charts']) == 1
    
    def test_charts_structure(self):
        """Test charts structure in results."""
        query = "sales metrics"
        candidates = [
            {
                'entity_id': 2,
                'title': 'Sales Dashboard',
                'url': 'https://superset.example.com/dashboard/2',
                'score': 0.88,
                'usage_score': 5.0,
                'signal_sources': ['trigram'],
                'matched_tokens': [],
                'charts': [
                    {
                        'chart_id': 201,
                        'title': 'Sales by Region',
                        'url': 'https://superset.example.com/chart/201',
                        'filters_default': {'region': 'all'}
                    },
                    {
                        'chart_id': 202,
                        'title': 'Sales Trend',
                        'url': 'https://superset.example.com/chart/202',
                        'filters_default': None
                    },
                    {
                        'chart_id': 203,
                        'title': 'Top Products',
                        'url': 'https://superset.example.com/chart/203',
                        'filters_default': {'limit': 10}
                    }
                ]
            }
        ]
        
        result = build_answer(query, candidates)
        charts = result['results'][0]['charts']
        
        # Should limit to max 3 charts
        assert len(charts) <= 3
        assert len(charts) == 3  # All 3 should be included
        
        # Check chart structure
        for chart in charts:
            assert 'title' in chart
            assert 'url' in chart
            assert isinstance(chart['title'], str)
            assert isinstance(chart['url'], str)
        
        # Check specific charts
        assert charts[0]['title'] == 'Sales by Region'
        assert charts[1]['title'] == 'Sales Trend'
        assert charts[2]['title'] == 'Top Products'
    
    def test_charts_limit_to_three(self):
        """Test that charts are limited to maximum 3 per result."""
        query = "test dashboard"
        candidates = [
            {
                'entity_id': 3,
                'title': 'Test Dashboard',
                'url': 'https://superset.example.com/dashboard/3',
                'score': 0.90,
                'usage_score': 0,
                'signal_sources': ['vector'],
                'matched_tokens': [],
                'charts': [
                    {'chart_id': i, 'title': f'Chart {i}', 'url': f'https://superset.example.com/chart/{i}', 'filters_default': None}
                    for i in range(301, 308)  # 7 charts
                ]
            }
        ]
        
        result = build_answer(query, candidates)
        charts = result['results'][0]['charts']
        
        # Should be limited to 3 charts
        assert len(charts) == 3
        assert charts[0]['title'] == 'Chart 301'
        assert charts[1]['title'] == 'Chart 302'
        assert charts[2]['title'] == 'Chart 303'
    
    def test_why_explanation_fts_match(self):
        """Test 'why' explanation for FTS matches."""
        query = "revenue analytics"
        candidates = [
            {
                'entity_id': 4,
                'title': 'Revenue Analytics Dashboard',
                'url': 'https://superset.example.com/dashboard/4',
                'score': 0.92,
                'usage_score': 0,
                'signal_sources': ['fts'],
                'matched_tokens': ['revenue', 'analytics', 'dashboard'],
                'charts': []
            }
        ]
        
        result = build_answer(query, candidates)
        why = result['results'][0]['why']
        
        assert 'совпадения в тексте:' in why
        assert 'revenue' in why
        assert 'analytics' in why
        assert 'dashboard' in why
    
    def test_why_explanation_vector_match(self):
        """Test 'why' explanation for vector similarity."""
        query = "user retention"
        candidates = [
            {
                'entity_id': 5,
                'title': 'User Engagement Dashboard',
                'url': 'https://superset.example.com/dashboard/5',
                'score': 0.87,
                'usage_score': 0,
                'signal_sources': ['vector'],
                'matched_tokens': [],
                'charts': []
            }
        ]
        
        result = build_answer(query, candidates)
        why = result['results'][0]['why']
        
        assert 'семантическая близость по содержимому' in why
    
    def test_why_explanation_trigram_match(self):
        """Test 'why' explanation for trigram matches."""
        query = "user dashboard"
        candidates = [
            {
                'entity_id': 6,
                'title': 'User Analytics Dashboard',
                'url': 'https://superset.example.com/dashboard/6',
                'score': 0.85,
                'usage_score': 0,
                'signal_sources': ['trigram'],
                'matched_tokens': [],
                'charts': []
            }
        ]
        
        result = build_answer(query, candidates)
        why = result['results'][0]['why']
        
        assert 'совпадение по заголовку' in why
    
    def test_why_explanation_with_popularity(self):
        """Test 'why' explanation includes popularity when usage_score > 0."""
        query = "popular dashboard"
        candidates = [
            {
                'entity_id': 7,
                'title': 'Popular Dashboard',
                'url': 'https://superset.example.com/dashboard/7',
                'score': 0.80,
                'usage_score': 15.5,
                'signal_sources': ['vector'],
                'matched_tokens': [],
                'charts': []
            }
        ]
        
        result = build_answer(query, candidates)
        why = result['results'][0]['why']
        
        assert 'семантическая близость по содержимому' in why
        assert 'популярность учтена' in why
    
    def test_why_explanation_combined_signals(self):
        """Test 'why' explanation for multiple signal sources."""
        query = "revenue dashboard analytics"
        candidates = [
            {
                'entity_id': 8,
                'title': 'Revenue Analytics Dashboard',
                'url': 'https://superset.example.com/dashboard/8',
                'score': 0.95,
                'usage_score': 8.2,
                'signal_sources': ['fts', 'vector', 'trigram'],
                'matched_tokens': ['revenue', 'analytics'],
                'charts': []
            }
        ]
        
        result = build_answer(query, candidates)
        why = result['results'][0]['why']
        
        assert 'совпадения в тексте:' in why
        assert 'revenue' in why
        assert 'analytics' in why
        assert 'семантическая близость по содержимому' in why
        assert 'совпадение по заголовку' in why
        assert 'популярность учтена' in why
    
    def test_suggested_filters_from_charts(self):
        """Test suggested_filters extraction from chart filters_default."""
        query = "sales data"
        candidates = [
            {
                'entity_id': 9,
                'title': 'Sales Dashboard',
                'url': 'https://superset.example.com/dashboard/9',
                'score': 0.90,
                'usage_score': 0,
                'signal_sources': ['vector'],
                'matched_tokens': [],
                'charts': [
                    {
                        'chart_id': 901,
                        'title': 'Sales by Region',
                        'url': 'https://superset.example.com/chart/901',
                        'filters_default': {'region': ['US', 'EU'], 'time_range': 'last_30_days'}
                    },
                    {
                        'chart_id': 902,
                        'title': 'Sales Trend',
                        'url': 'https://superset.example.com/chart/902',
                        'filters_default': {'time_range': 'last_90_days', 'product_type': 'all'}
                    },
                    {
                        'chart_id': 903,
                        'title': 'Top Products',
                        'url': 'https://superset.example.com/chart/903',
                        'filters_default': None
                    }
                ]
            }
        ]
        
        result = build_answer(query, candidates)
        suggested_filters = result['suggested_filters']
        
        # Should collect unique filter keys from all charts
        assert isinstance(suggested_filters, dict)
        assert 'region' in suggested_filters
        assert 'time_range' in suggested_filters
        assert 'product_type' in suggested_filters
        
        # Should merge values from different charts
        assert suggested_filters['region'] == ['US', 'EU']
        # time_range appears in multiple charts, should take first occurrence
        assert suggested_filters['time_range'] == 'last_30_days'
        assert suggested_filters['product_type'] == 'all'
    
    def test_suggested_filters_empty_when_no_filters(self):
        """Test suggested_filters is empty when no charts have filters_default."""
        query = "simple dashboard"
        candidates = [
            {
                'entity_id': 10,
                'title': 'Simple Dashboard',
                'url': 'https://superset.example.com/dashboard/10',
                'score': 0.75,
                'usage_score': 0,
                'signal_sources': ['vector'],
                'matched_tokens': [],
                'charts': [
                    {
                        'chart_id': 1001,
                        'title': 'Simple Chart',
                        'url': 'https://superset.example.com/chart/1001',
                        'filters_default': None
                    }
                ]
            }
        ]
        
        result = build_answer(query, candidates)
        suggested_filters = result['suggested_filters']
        
        assert isinstance(suggested_filters, dict)
        assert len(suggested_filters) == 0
    
    def test_multiple_candidates(self):
        """Test handling multiple candidates."""
        query = "analytics"
        candidates = [
            {
                'entity_id': 11,
                'title': 'Revenue Analytics',
                'url': 'https://superset.example.com/dashboard/11',
                'score': 0.95,
                'usage_score': 10.0,
                'signal_sources': ['fts'],
                'matched_tokens': ['analytics'],
                'charts': []
            },
            {
                'entity_id': 12,
                'title': 'User Analytics',
                'url': 'https://superset.example.com/dashboard/12',
                'score': 0.88,
                'usage_score': 5.0,
                'signal_sources': ['vector'],
                'matched_tokens': [],
                'charts': []
            }
        ]
        
        result = build_answer(query, candidates)
        
        assert len(result['results']) == 2
        assert result['results'][0]['title'] == 'Revenue Analytics'
        assert result['results'][1]['title'] == 'User Analytics'
        
        # Check different explanations
        assert 'совпадения в тексте:' in result['results'][0]['why']
        assert 'семантическая близость по содержимому' in result['results'][1]['why']
    
    def test_empty_candidates(self):
        """Test handling empty candidates list."""
        query = "nonexistent dashboard"
        candidates = []
        
        result = build_answer(query, candidates)
        
        assert isinstance(result, dict)
        assert 'results' in result
        assert 'suggested_filters' in result
        assert result['results'] == []
        assert result['suggested_filters'] == {}


class TestAnswerBuilderClass:
    """Test AnswerBuilder class methods."""
    
    def test_build_why_explanation_fts_only(self):
        """Test building why explanation for FTS only."""
        builder = AnswerBuilder()
        
        candidate = {
            'signal_sources': ['fts'],
            'matched_tokens': ['revenue', 'dashboard'],
            'usage_score': 0
        }
        
        why = builder._build_why_explanation(candidate)
        
        assert why == 'совпадения в тексте: revenue, dashboard'
    
    def test_build_why_explanation_vector_only(self):
        """Test building why explanation for vector only."""
        builder = AnswerBuilder()
        
        candidate = {
            'signal_sources': ['vector'],
            'matched_tokens': [],
            'usage_score': 0
        }
        
        why = builder._build_why_explanation(candidate)
        
        assert why == 'семантическая близость по содержимому'
    
    def test_build_why_explanation_trigram_only(self):
        """Test building why explanation for trigram only."""
        builder = AnswerBuilder()
        
        candidate = {
            'signal_sources': ['trigram'],
            'matched_tokens': [],
            'usage_score': 0
        }
        
        why = builder._build_why_explanation(candidate)
        
        assert why == 'совпадение по заголовку'
    
    def test_build_why_explanation_with_popularity(self):
        """Test building why explanation with popularity."""
        builder = AnswerBuilder()
        
        candidate = {
            'signal_sources': ['vector'],
            'matched_tokens': [],
            'usage_score': 10.5
        }
        
        why = builder._build_why_explanation(candidate)
        
        assert why == 'семантическая близость по содержимому; популярность учтена'
    
    def test_extract_suggested_filters(self):
        """Test extracting suggested filters from candidates."""
        builder = AnswerBuilder()
        
        candidates = [
            {
                'charts': [
                    {
                        'filters_default': {'region': 'US', 'time_range': 'last_30_days'}
                    },
                    {
                        'filters_default': {'region': 'EU', 'product_type': 'premium'}
                    }
                ]
            },
            {
                'charts': [
                    {
                        'filters_default': {'time_range': 'last_90_days', 'category': 'sales'}
                    },
                    {
                        'filters_default': None
                    }
                ]
            }
        ]
        
        filters = builder._extract_suggested_filters(candidates)
        
        # Should collect all unique filter keys
        assert 'region' in filters
        assert 'time_range' in filters
        assert 'product_type' in filters
        assert 'category' in filters
        
        # Should take first occurrence of each key
        assert filters['region'] == 'US'
        assert filters['time_range'] == 'last_30_days'
        assert filters['product_type'] == 'premium'
        assert filters['category'] == 'sales'
    
    def test_extract_suggested_filters_empty(self):
        """Test extracting suggested filters when none exist."""
        builder = AnswerBuilder()
        
        candidates = [
            {
                'charts': [
                    {'filters_default': None},
                    {'filters_default': {}}
                ]
            }
        ]
        
        filters = builder._extract_suggested_filters(candidates)
        
        assert filters == {}
