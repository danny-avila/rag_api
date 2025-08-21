# app/dash_assistant/serving/answer_builder.py
"""Answer builder for dash assistant search results."""
from typing import Dict, List, Any, Optional
from app.config import logger


class AnswerBuilder:
    """Builder for structured answers from search candidates."""
    
    def build_answer(self, query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build structured answer from search candidates.
        
        Args:
            query: User search query
            candidates: List of search candidates with diagnostics
            
        Returns:
            Dict containing structured answer with results and suggested_filters
        """
        logger.debug(f"Building answer for query: '{query}' with {len(candidates)} candidates")
        
        # Build results from candidates
        results = []
        for candidate in candidates:
            result_item = self._build_result_item(candidate)
            results.append(result_item)
        
        # Extract suggested filters from all candidates
        suggested_filters = self._extract_suggested_filters(candidates)
        
        answer = {
            'results': results,
            'suggested_filters': suggested_filters
        }
        
        logger.debug(f"Built answer with {len(results)} results and {len(suggested_filters)} suggested filters")
        return answer
    
    def _build_result_item(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Build a single result item from candidate.
        
        Args:
            candidate: Search candidate with metadata
            
        Returns:
            Dict containing structured result item
        """
        # Extract basic fields
        result_item = {
            'title': candidate.get('title', ''),
            'url': candidate.get('url', ''),
            'score': candidate.get('score', 0.0)
        }
        
        # Process charts (limit to 3)
        charts = candidate.get('charts', [])
        processed_charts = []
        for chart in charts[:3]:  # Limit to maximum 3 charts
            chart_item = {
                'title': chart.get('title', ''),
                'url': chart.get('url', '')
            }
            processed_charts.append(chart_item)
        
        result_item['charts'] = processed_charts
        
        # Build explanation
        result_item['why'] = self._build_why_explanation(candidate)
        
        return result_item
    
    def _build_why_explanation(self, candidate: Dict[str, Any]) -> str:
        """Build explanation for why this candidate was selected.
        
        Args:
            candidate: Search candidate with signal sources and metadata
            
        Returns:
            String explanation of selection reasoning
        """
        signal_sources = candidate.get('signal_sources', [])
        matched_tokens = candidate.get('matched_tokens', [])
        usage_score = candidate.get('usage_score', 0)
        
        explanations = []
        
        # Check for FTS matches
        if 'fts' in signal_sources and matched_tokens:
            # Show top matched tokens (limit to reasonable number)
            top_tokens = matched_tokens[:5]  # Show max 5 tokens
            tokens_str = ', '.join(top_tokens)
            explanations.append(f'совпадения в тексте: {tokens_str}')
        
        # Check for vector similarity
        if 'vector' in signal_sources:
            explanations.append('семантическая близость по содержимому')
        
        # Check for trigram matches
        if 'trigram' in signal_sources:
            explanations.append('совпадение по заголовку')
        
        # Add popularity if usage_score > 0
        if usage_score > 0:
            explanations.append('популярность учтена')
        
        # Join explanations
        if explanations:
            return '; '.join(explanations)
        else:
            return 'найдено по общему соответствию'
    
    def _extract_suggested_filters(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract suggested filters from candidates' charts.
        
        Args:
            candidates: List of search candidates
            
        Returns:
            Dict with suggested filter keys and values
        """
        suggested_filters = {}
        
        for candidate in candidates:
            charts = candidate.get('charts', [])
            
            for chart in charts:
                filters_default = chart.get('filters_default')
                
                if filters_default and isinstance(filters_default, dict):
                    for key, value in filters_default.items():
                        # Only add if we haven't seen this filter key before
                        if key not in suggested_filters:
                            suggested_filters[key] = value
        
        return suggested_filters


def build_answer(query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convenience function to build answer from search candidates.
    
    Args:
        query: User search query
        candidates: List of search candidates with diagnostics
        
    Returns:
        Dict containing structured answer with results and suggested_filters
    """
    builder = AnswerBuilder()
    return builder.build_answer(query, candidates)
