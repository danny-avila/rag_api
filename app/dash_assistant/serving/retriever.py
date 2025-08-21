# app/dash_assistant/serving/retriever.py
"""Retriever for dash assistant using RRF (Reciprocal Rank Fusion)."""
import asyncio
from typing import Dict, List, Any, Optional
from app.config import logger
from app.dash_assistant.db import DashAssistantDB
from app.dash_assistant.indexing.embedder import get_embedder
from app.dash_assistant.config import get_config


class DashRetriever:
    """Retriever for dashboard search using multiple signals and RRF."""
    
    def __init__(self):
        """Initialize retriever."""
        self.db = DashAssistantDB
        self.embedder = get_embedder()
        logger.info("Initialized DashRetriever with RRF")
    
    async def search(self, 
                    query: str, 
                    top_k: Optional[int] = None, 
                    filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for dashboards using multiple signals and RRF.
        
        Args:
            query: Search query
            top_k: Number of results to return (uses config default if None)
            filters: Optional filters to apply
            
        Returns:
            List of candidates with scores and metadata
        """
        config = get_config()
        if top_k is None:
            top_k = config.default_topk
            
        logger.debug(f"Searching for: '{query}' with top_k={top_k}, filters={filters}")
        
        # Run multiple search methods in parallel
        fts_task = self._search_fts(query, top_k * 2, filters)
        vector_task = self._search_vector(query, top_k * 2, filters)
        trigram_task = self._search_trigram(query, top_k * 2, filters)
        
        fts_results, vector_results, trigram_results = await asyncio.gather(
            fts_task, vector_task, trigram_task, return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(fts_results, Exception):
            logger.warning(f"FTS search failed: {fts_results}")
            fts_results = []
        
        if isinstance(vector_results, Exception):
            logger.warning(f"Vector search failed: {vector_results}")
            vector_results = []
        
        if isinstance(trigram_results, Exception):
            logger.warning(f"Trigram search failed: {trigram_results}")
            trigram_results = []
        
        # Apply RRF to combine results
        combined_results = self._apply_rrf(
            fts_results=fts_results,
            vector_results=vector_results,
            trigram_results=trigram_results,
            top_k=top_k
        )
        
        # Enrich results with charts and metadata
        enriched_results = await self._enrich_results(combined_results)
        
        logger.debug(f"Found {len(enriched_results)} results after RRF")
        return enriched_results
    
    async def _search_fts(self, 
                         query: str, 
                         limit: int, 
                         filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search using Full Text Search."""
        try:
            # Build WHERE clause for filters
            where_conditions = ["1=1"]  # Base condition
            params = [query]
            param_index = 2
            
            if filters:
                for key, value in filters.items():
                    if key in ['domain', 'owner']:
                        where_conditions.append(f"e.{key} = ${param_index}")
                        params.append(value)
                        param_index += 1
            
            where_clause = " AND ".join(where_conditions)
            
            # FTS query with ranking
            query_sql = f"""
                SELECT DISTINCT
                    e.entity_id,
                    e.title,
                    e.url,
                    e.usage_score,
                    ts_rank(c.tsv_en, plainto_tsquery('english', $1)) as fts_score,
                    array_agg(DISTINCT word) as matched_tokens
                FROM bi_entity e
                JOIN bi_chunk c ON e.entity_id = c.entity_id
                CROSS JOIN LATERAL unnest(string_to_array(lower($1), ' ')) as word
                WHERE c.tsv_en @@ plainto_tsquery('english', $1)
                  AND {where_clause}
                GROUP BY e.entity_id, e.title, e.url, e.usage_score, c.tsv_en
                ORDER BY fts_score DESC, e.usage_score DESC
                LIMIT {limit}
            """
            
            results = await self.db.fetch_all(query_sql, *params)
            
            fts_results = []
            for i, row in enumerate(results):
                fts_results.append({
                    'entity_id': row['entity_id'],
                    'title': row['title'],
                    'url': row['url'],
                    'usage_score': row['usage_score'] or 0,
                    'fts_score': float(row['fts_score']),
                    'fts_rank': i + 1,
                    'matched_tokens': row['matched_tokens'] or [],
                    'signal_sources': ['fts']
                })
            
            logger.debug(f"FTS search found {len(fts_results)} results")
            return fts_results
            
        except Exception as e:
            logger.error(f"FTS search failed: {e}")
            return []
    
    async def _search_vector(self, 
                           query: str, 
                           limit: int, 
                           filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search using vector similarity."""
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_text(query)
            
            # Build WHERE clause for filters
            where_conditions = ["c.embedding IS NOT NULL"]
            params = [query_embedding.tolist()]
            param_index = 2
            
            if filters:
                for key, value in filters.items():
                    if key in ['domain', 'owner']:
                        where_conditions.append(f"e.{key} = ${param_index}")
                        params.append(value)
                        param_index += 1
            
            where_clause = " AND ".join(where_conditions)
            
            # Vector similarity query
            query_sql = f"""
                SELECT DISTINCT
                    e.entity_id,
                    e.title,
                    e.url,
                    e.usage_score,
                    1 - (c.embedding <=> $1) as vector_score
                FROM bi_entity e
                JOIN bi_chunk c ON e.entity_id = c.entity_id
                WHERE {where_clause}
                ORDER BY c.embedding <=> $1, e.usage_score DESC
                LIMIT {limit}
            """
            
            results = await self.db.fetch_all(query_sql, *params)
            
            vector_results = []
            for i, row in enumerate(results):
                vector_results.append({
                    'entity_id': row['entity_id'],
                    'title': row['title'],
                    'url': row['url'],
                    'usage_score': row['usage_score'] or 0,
                    'vector_score': float(row['vector_score']),
                    'vector_rank': i + 1,
                    'matched_tokens': [],
                    'signal_sources': ['vector']
                })
            
            logger.debug(f"Vector search found {len(vector_results)} results")
            return vector_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def _search_trigram(self, 
                            query: str, 
                            limit: int, 
                            filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search using trigram similarity on titles."""
        try:
            # Build WHERE clause for filters
            where_conditions = ["1=1"]
            params = [query]
            param_index = 2
            
            if filters:
                for key, value in filters.items():
                    if key in ['domain', 'owner']:
                        where_conditions.append(f"e.{key} = ${param_index}")
                        params.append(value)
                        param_index += 1
            
            where_clause = " AND ".join(where_conditions)
            
            # Trigram similarity query
            query_sql = f"""
                SELECT DISTINCT
                    e.entity_id,
                    e.title,
                    e.url,
                    e.usage_score,
                    similarity(e.title, $1) as trigram_score
                FROM bi_entity e
                WHERE similarity(e.title, $1) > 0.1
                  AND {where_clause}
                ORDER BY trigram_score DESC, e.usage_score DESC
                LIMIT {limit}
            """
            
            results = await self.db.fetch_all(query_sql, *params)
            
            trigram_results = []
            for i, row in enumerate(results):
                trigram_results.append({
                    'entity_id': row['entity_id'],
                    'title': row['title'],
                    'url': row['url'],
                    'usage_score': row['usage_score'] or 0,
                    'trigram_score': float(row['trigram_score']),
                    'trigram_rank': i + 1,
                    'matched_tokens': [],
                    'signal_sources': ['trigram']
                })
            
            logger.debug(f"Trigram search found {len(trigram_results)} results")
            return trigram_results
            
        except Exception as e:
            logger.error(f"Trigram search failed: {e}")
            return []
    
    def _apply_rrf(self, 
                   fts_results: List[Dict[str, Any]], 
                   vector_results: List[Dict[str, Any]], 
                   trigram_results: List[Dict[str, Any]], 
                   top_k: int, 
                   k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Apply Reciprocal Rank Fusion to combine results.
        
        Args:
            fts_results: FTS search results
            vector_results: Vector search results
            trigram_results: Trigram search results
            top_k: Number of final results to return
            k: RRF parameter (uses config default if None)
            
        Returns:
            Combined and ranked results
        """
        config = get_config()
        if k is None:
            k = config.rrf_k
            
        # Collect all unique entities
        entity_scores = {}
        
        # Process FTS results
        for result in fts_results:
            entity_id = result['entity_id']
            if entity_id not in entity_scores:
                entity_scores[entity_id] = {
                    'entity_id': entity_id,
                    'title': result['title'],
                    'url': result['url'],
                    'usage_score': result['usage_score'],
                    'rrf_score': 0.0,
                    'signal_sources': set(),
                    'matched_tokens': set(),
                    'individual_scores': {}
                }
            
            # Add RRF score contribution
            rrf_contribution = 1.0 / (k + result['fts_rank'])
            entity_scores[entity_id]['rrf_score'] += rrf_contribution
            entity_scores[entity_id]['signal_sources'].add('fts')
            entity_scores[entity_id]['matched_tokens'].update(result['matched_tokens'])
            entity_scores[entity_id]['individual_scores']['fts'] = result['fts_score']
        
        # Process Vector results
        for result in vector_results:
            entity_id = result['entity_id']
            if entity_id not in entity_scores:
                entity_scores[entity_id] = {
                    'entity_id': entity_id,
                    'title': result['title'],
                    'url': result['url'],
                    'usage_score': result['usage_score'],
                    'rrf_score': 0.0,
                    'signal_sources': set(),
                    'matched_tokens': set(),
                    'individual_scores': {}
                }
            
            # Add RRF score contribution
            rrf_contribution = 1.0 / (k + result['vector_rank'])
            entity_scores[entity_id]['rrf_score'] += rrf_contribution
            entity_scores[entity_id]['signal_sources'].add('vector')
            entity_scores[entity_id]['individual_scores']['vector'] = result['vector_score']
        
        # Process Trigram results
        for result in trigram_results:
            entity_id = result['entity_id']
            if entity_id not in entity_scores:
                entity_scores[entity_id] = {
                    'entity_id': entity_id,
                    'title': result['title'],
                    'url': result['url'],
                    'usage_score': result['usage_score'],
                    'rrf_score': 0.0,
                    'signal_sources': set(),
                    'matched_tokens': set(),
                    'individual_scores': {}
                }
            
            # Add RRF score contribution
            rrf_contribution = 1.0 / (k + result['trigram_rank'])
            entity_scores[entity_id]['rrf_score'] += rrf_contribution
            entity_scores[entity_id]['signal_sources'].add('trigram')
            entity_scores[entity_id]['individual_scores']['trigram'] = result['trigram_score']
        
        # Add popularity boost
        for entity_id, entity_data in entity_scores.items():
            usage_score = entity_data['usage_score']
            if usage_score > 0:
                # Add small popularity boost (normalized)
                popularity_boost = min(usage_score / 100.0, 0.1)  # Max 0.1 boost
                entity_data['rrf_score'] += popularity_boost
        
        # Convert sets to lists and sort by RRF score
        final_results = []
        for entity_data in entity_scores.values():
            entity_data['signal_sources'] = list(entity_data['signal_sources'])
            entity_data['matched_tokens'] = list(entity_data['matched_tokens'])
            entity_data['score'] = entity_data['rrf_score']  # Final score
            final_results.append(entity_data)
        
        # Sort by RRF score (descending) and return top_k
        final_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        
        logger.debug(f"RRF combined {len(entity_scores)} unique entities, returning top {top_k}")
        return final_results[:top_k]
    
    async def _enrich_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich results with charts and additional metadata.
        
        Args:
            results: Basic search results
            
        Returns:
            Enriched results with charts
        """
        if not results:
            return results
        
        # Get entity IDs
        entity_ids = [result['entity_id'] for result in results]
        
        # Fetch charts for all entities in one query
        charts_query = """
            SELECT 
                c.parent_dashboard_id as entity_id,
                c.chart_id,
                c.title,
                c.url,
                c.filters_default
            FROM bi_chart c
            WHERE c.parent_dashboard_id = ANY($1)
            ORDER BY c.parent_dashboard_id, c.chart_id
        """
        
        charts_results = await self.db.fetch_all(charts_query, entity_ids)
        
        # Group charts by entity_id
        charts_by_entity = {}
        for chart in charts_results:
            entity_id = chart['entity_id']
            if entity_id not in charts_by_entity:
                charts_by_entity[entity_id] = []
            
            chart_data = {
                'chart_id': chart['chart_id'],
                'title': chart['title'],
                'url': chart['url'],
                'filters_default': chart['filters_default']
            }
            charts_by_entity[entity_id].append(chart_data)
        
        # Add charts to results
        for result in results:
            entity_id = result['entity_id']
            result['charts'] = charts_by_entity.get(entity_id, [])
        
        return results
