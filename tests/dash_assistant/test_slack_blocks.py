# tests/dash_assistant/test_slack_blocks.py
"""Tests for Slack blocks functionality."""
import pytest
from typing import Dict, List, Any
from app.dash_assistant.slack.blocks import build_slack_blocks


class TestSlackBlocks:
    """Test Slack blocks builder functionality."""
    
    def test_build_slack_blocks_basic_structure(self):
        """Test basic structure of Slack blocks."""
        answer = {
            'results': [
                {
                    'title': 'Revenue Analytics Dashboard',
                    'url': 'https://superset.example.com/dashboard/1',
                    'score': 0.95,
                    'why': 'совпадения в тексте: revenue, analytics; популярность учтена',
                    'charts': [
                        {
                            'title': 'Monthly Revenue Trend',
                            'url': 'https://superset.example.com/chart/101'
                        }
                    ]
                }
            ],
            'suggested_filters': {'time_range': 'last_30_days'}
        }
        
        blocks = build_slack_blocks(answer)
        
        # Check basic structure
        assert isinstance(blocks, list)
        assert len(blocks) >= 2  # At least header + one result
        
        # Check header block
        header_block = blocks[0]
        assert header_block['type'] == 'header'
        assert 'text' in header_block
        assert header_block['text']['type'] == 'plain_text'
        assert 'Найденные дашборды' in header_block['text']['text']
    
    def test_build_slack_blocks_single_result(self):
        """Test Slack blocks for single result."""
        answer = {
            'results': [
                {
                    'title': 'User Retention Dashboard',
                    'url': 'https://superset.example.com/dashboard/2',
                    'score': 0.88,
                    'why': 'семантическая близость по содержимому',
                    'charts': [
                        {
                            'title': 'Retention Cohort Analysis',
                            'url': 'https://superset.example.com/chart/201'
                        },
                        {
                            'title': 'User Churn Rate',
                            'url': 'https://superset.example.com/chart/202'
                        }
                    ]
                }
            ],
            'suggested_filters': {}
        }
        
        blocks = build_slack_blocks(answer)
        
        # Should have header + result section + actions
        assert len(blocks) >= 3
        
        # Find result section
        result_section = None
        for block in blocks:
            if block.get('type') == 'section' and 'fields' in block:
                result_section = block
                break
        
        assert result_section is not None
        
        # Check result section structure
        assert 'fields' in result_section
        fields = result_section['fields']
        
        # Should have title, URL, why, charts fields
        field_texts = [field['text'] for field in fields]
        combined_text = ' '.join(field_texts)
        
        assert 'User Retention Dashboard' in combined_text
        assert 'https://superset.example.com/dashboard/2' in combined_text
        assert 'семантическая близость по содержимому' in combined_text
        assert 'Retention Cohort Analysis' in combined_text
        assert 'User Churn Rate' in combined_text
    
    def test_build_slack_blocks_actions(self):
        """Test action buttons in Slack blocks."""
        answer = {
            'results': [
                {
                    'title': 'Sales Dashboard',
                    'url': 'https://superset.example.com/dashboard/3',
                    'score': 0.75,
                    'why': 'совпадение по заголовку',
                    'charts': []
                }
            ],
            'suggested_filters': {}
        }
        
        blocks = build_slack_blocks(answer)
        
        # Find actions block
        actions_block = None
        for block in blocks:
            if block.get('type') == 'actions':
                actions_block = block
                break
        
        assert actions_block is not None
        assert 'elements' in actions_block
        
        elements = actions_block['elements']
        assert len(elements) == 3  # "Открыть", "👍", "👎"
        
        # Check "Открыть" button
        open_button = elements[0]
        assert open_button['type'] == 'button'
        assert open_button['text']['text'] == 'Открыть'
        assert open_button['url'] == 'https://superset.example.com/dashboard/3'
        assert open_button['style'] == 'primary'
        
        # Check "👍" button
        thumbs_up = elements[1]
        assert thumbs_up['type'] == 'button'
        assert thumbs_up['text']['text'] == '👍'
        assert thumbs_up['value'] == 'up'
        
        # Check "👎" button
        thumbs_down = elements[2]
        assert thumbs_down['type'] == 'button'
        assert thumbs_down['text']['text'] == '👎'
        assert thumbs_down['value'] == 'down'
    
    def test_build_slack_blocks_multiple_results(self):
        """Test Slack blocks for multiple results."""
        answer = {
            'results': [
                {
                    'title': 'Revenue Analytics',
                    'url': 'https://superset.example.com/dashboard/1',
                    'score': 0.95,
                    'why': 'совпадения в тексте: revenue; популярность учтена',
                    'charts': [{'title': 'Revenue Chart', 'url': 'https://superset.example.com/chart/1'}]
                },
                {
                    'title': 'Sales Analytics',
                    'url': 'https://superset.example.com/dashboard/2',
                    'score': 0.85,
                    'why': 'семантическая близость по содержимому',
                    'charts': []
                }
            ],
            'suggested_filters': {'region': 'US'}
        }
        
        blocks = build_slack_blocks(answer)
        
        # Should have header + 2 result sections + 2 action blocks + dividers
        assert len(blocks) >= 6
        
        # Count result sections
        result_sections = [block for block in blocks if block.get('type') == 'section' and 'fields' in block]
        assert len(result_sections) == 2
        
        # Count action blocks
        action_blocks = [block for block in blocks if block.get('type') == 'actions']
        assert len(action_blocks) == 2
        
        # Check that each result has corresponding actions
        for i, result in enumerate(answer['results']):
            # Find corresponding action block
            action_block = action_blocks[i]
            open_button = action_block['elements'][0]
            assert open_button['url'] == result['url']
    
    def test_build_slack_blocks_no_results(self):
        """Test Slack blocks when no results found."""
        answer = {
            'results': [],
            'suggested_filters': {}
        }
        
        blocks = build_slack_blocks(answer)
        
        # Should have header + no results message
        assert len(blocks) == 2
        
        # Check header
        assert blocks[0]['type'] == 'header'
        
        # Check no results message
        no_results_block = blocks[1]
        assert no_results_block['type'] == 'section'
        assert 'text' in no_results_block
        assert 'Дашборды не найдены' in no_results_block['text']['text']
    
    def test_build_slack_blocks_with_suggested_filters(self):
        """Test Slack blocks with suggested filters."""
        answer = {
            'results': [
                {
                    'title': 'Test Dashboard',
                    'url': 'https://superset.example.com/dashboard/1',
                    'score': 0.80,
                    'why': 'найдено по общему соответствию',
                    'charts': []
                }
            ],
            'suggested_filters': {
                'time_range': 'last_30_days',
                'region': 'US',
                'product_type': 'premium'
            }
        }
        
        blocks = build_slack_blocks(answer)
        
        # Find suggested filters section
        filters_section = None
        for block in blocks:
            if (block.get('type') == 'section' and 
                'text' in block and 
                'Рекомендуемые фильтры' in block['text']['text']):
                filters_section = block
                break
        
        assert filters_section is not None
        
        # Check filters are mentioned
        filters_text = filters_section['text']['text']
        assert 'time_range: last_30_days' in filters_text
        assert 'region: US' in filters_text
        assert 'product_type: premium' in filters_text
    
    def test_build_slack_blocks_long_title_truncation(self):
        """Test that long titles are properly handled."""
        answer = {
            'results': [
                {
                    'title': 'Very Long Dashboard Title That Exceeds Normal Length And Should Be Handled Properly Without Breaking The Layout',
                    'url': 'https://superset.example.com/dashboard/1',
                    'score': 0.90,
                    'why': 'совпадения в тексте: long, title',
                    'charts': []
                }
            ],
            'suggested_filters': {}
        }
        
        blocks = build_slack_blocks(answer)
        
        # Should still create valid blocks
        assert isinstance(blocks, list)
        assert len(blocks) >= 3
        
        # Find result section
        result_section = None
        for block in blocks:
            if block.get('type') == 'section' and 'fields' in block:
                result_section = block
                break
        
        assert result_section is not None
        
        # Check that title is included (may be truncated but should be present)
        fields_text = ' '.join([field['text'] for field in result_section['fields']])
        assert 'Very Long Dashboard Title' in fields_text
    
    def test_build_slack_blocks_charts_formatting(self):
        """Test proper formatting of charts in blocks."""
        answer = {
            'results': [
                {
                    'title': 'Analytics Dashboard',
                    'url': 'https://superset.example.com/dashboard/1',
                    'score': 0.92,
                    'why': 'семантическая близость по содержимому',
                    'charts': [
                        {
                            'title': 'Revenue Trend',
                            'url': 'https://superset.example.com/chart/1'
                        },
                        {
                            'title': 'User Acquisition',
                            'url': 'https://superset.example.com/chart/2'
                        },
                        {
                            'title': 'Conversion Funnel',
                            'url': 'https://superset.example.com/chart/3'
                        }
                    ]
                }
            ],
            'suggested_filters': {}
        }
        
        blocks = build_slack_blocks(answer)
        
        # Find result section
        result_section = None
        for block in blocks:
            if block.get('type') == 'section' and 'fields' in block:
                result_section = block
                break
        
        assert result_section is not None
        
        # Check charts formatting
        fields_text = ' '.join([field['text'] for field in result_section['fields']])
        
        # All chart titles should be present
        assert 'Revenue Trend' in fields_text
        assert 'User Acquisition' in fields_text
        assert 'Conversion Funnel' in fields_text
        
        # Should have proper formatting (links or bullet points)
        assert '•' in fields_text or '<' in fields_text  # Either bullets or Slack links
    
    def test_build_slack_blocks_special_characters(self):
        """Test handling of special characters in text."""
        answer = {
            'results': [
                {
                    'title': 'Dashboard with "Quotes" & <Special> Characters',
                    'url': 'https://superset.example.com/dashboard/1',
                    'score': 0.85,
                    'why': 'совпадения в тексте: special, characters',
                    'charts': [
                        {
                            'title': 'Chart with & symbols',
                            'url': 'https://superset.example.com/chart/1'
                        }
                    ]
                }
            ],
            'suggested_filters': {}
        }
        
        blocks = build_slack_blocks(answer)
        
        # Should create valid blocks without errors
        assert isinstance(blocks, list)
        assert len(blocks) >= 3
        
        # Find result section
        result_section = None
        for block in blocks:
            if block.get('type') == 'section' and 'fields' in block:
                result_section = block
                break
        
        assert result_section is not None
        
        # Check that special characters are handled
        fields_text = ' '.join([field['text'] for field in result_section['fields']])
        assert 'Dashboard with' in fields_text
        assert 'Chart with' in fields_text
