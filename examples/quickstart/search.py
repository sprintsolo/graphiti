"""
Copyright 2025, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import *
from graphiti_core.search.search_filters import SearchFilters  # Added import
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.search.search_config import (
    SearchConfig, 
    NodeSearchConfig, 
    NodeSearchMethod, 
    NodeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    EdgeReranker
)

#################################################
# CONFIGURATION
#################################################
# Set up logging and environment variables for
# connecting to Neo4j database
#################################################

# Configure logging
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

# Neo4j connection parameters
# Make sure Neo4j Desktop is running with a local DBMS started
neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
api_key = os.environ.get("LLM_API_KEY")
llm_model = os.environ.get("LLM_MODEL", "gemini-2.5-flash-preview-05-20")
embedding_model = os.environ.get("EMBEDDING_MODEL", "text-multilingual-embedding-002")
embedding_dim = int(os.environ.get("EMBEDDING_DIM", "768"))

if not api_key:
    raise ValueError("LLM_API_KEY environment variable must be set")

if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

async def get_episode_by_uuid(graphiti, uuid):
    """
    Retrieve an episode node by UUID
    
    Args:
        graphiti: The Graphiti instance
        uuid: UUID of the episode to retrieve
        
    Returns:
        The episode node if found, None otherwise
    """
    try:
        # Use the driver from the graphiti instance
        records, _, _ = await graphiti.driver.execute_query(
            """
            MATCH (e:Episodic {uuid: $uuid})
            RETURN 
                e.uuid AS uuid,
                e.name AS name,
                e.content AS content,
                e.source AS source,
                e.source_description AS source_description
            """,
            uuid=uuid,
        )
        
        if not records:
            logger.warning(f"No episode found with UUID: {uuid}")
            return None
            
        record = records[0]
        
        # Parse JSON content if it's a JSON source
        content = record['content']
        if record['source'] == 'json':
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON content for episode UUID: {uuid}")
        
        episode = {
            'uuid': record['uuid'],
            'name': record['name'],
            'content': content,
            'source': record['source'],
            'source_description': record['source_description']
        }
        
        logger.info(f"Retrieved episode: {record['name']} (UUID: {uuid})")
        return episode
    
    except Exception as e:
        logger.error(f"Error retrieving episode with UUID {uuid}: {str(e)}")
        return None



async def main():
    # Get search query from user input
    search_query = input("Enter your search query: ")
    print(f"\nUsing search query: '{search_query}'")

    #################################################
    # INITIALIZATION
    #################################################
    # Connect to Neo4j and set up Graphiti indices
    # This is required before using other Graphiti
    # functionality
    #################################################

    # Initialize Graphiti with Neo4j connection
    graphiti = Graphiti(
    neo4j_uri,
    neo4j_user,
    neo4j_password,
    llm_client=GeminiClient(
        config=LLMConfig(
            api_key=api_key,
            model=llm_model
        )
    ),
    embedder=GeminiEmbedder(
        config=GeminiEmbedderConfig(
            api_key=api_key,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim
        )
    ),
    cross_encoder=GeminiRerankerClient(
        config=LLMConfig(
            api_key=api_key,
            model=llm_model
        )
    )
)


    try:
        # 사용자 정의 검색 설정 만들기
        CUSTOM_SEARCH_CONFIG = SearchConfig(
            # 노드 검색 설정
            node_config=NodeSearchConfig(
                search_methods=[
                    NodeSearchMethod.bm25,               # 전체 텍스트 검색
                    NodeSearchMethod.cosine_similarity,  # 벡터 유사도 검색
                    #NodeSearchMethod.bfs               # 주석 해제하여 그래프 탐색 추가
                ],
                reranker=NodeReranker.rrf,     # 재순위 방법: cross_encoder, rrf, mmr, node_distance, episode_mentions
                sim_min_score=0.3,                       # 유사도 최소 점수
                bfs_max_depth=2
            ),
            # 엣지 검색 설정 (필요시 주석 해제)
            edge_config=EdgeSearchConfig(
                search_methods=[
                    #EdgeSearchMethod.bm25, 
                    EdgeSearchMethod.cosine_similarity
                ],
                reranker=EdgeReranker.rrf
            ),
            limit=5,                                    # 반환할 결과 수
            reranker_min_score=0.1                       # 재순위 최소 점수
        )
        
        # Create a search filter to only show Person nodes
        search_filter = SearchFilters()
        search_filter.node_labels = ["Company"]
        
        # Call search_ with the filter
        results = await graphiti.search_(
            search_query, 
            search_filter=search_filter,
            #center_node_uuid="7f191b74-54cc-4ca2-8770-ee78414a19bb",
            #bfs_origin_node_uuids=["7f191b74-54cc-4ca2-8770-ee78414a19bb"],
            config=CUSTOM_SEARCH_CONFIG  # 커스텀 설정 사용
        )

        # Print search results
        print('\nSearch Results:')
        
        # Handle SearchResults object from search_ method
        if hasattr(results, "edges") or hasattr(results, "nodes"):
            # Display edges
            if hasattr(results, "edges") and results.edges:
                print(f"\n=== EDGES ({len(results.edges)}) ===")
                for edge in results.edges:
                    print(f"UUID: {edge.uuid}")
                    print(f"Fact: {edge.fact}")
                    print(f"Source: {edge.source_node_uuid} -> Target: {edge.target_node_uuid}")
                    if hasattr(edge, 'valid_at') and edge.valid_at:
                        print(f"Valid from: {edge.valid_at}")       
                    if hasattr(edge, 'invalid_at') and edge.invalid_at:
                        print(f"Valid until: {edge.invalid_at}")
                    print("---")
            
            # Display nodes
            if hasattr(results, "nodes") and results.nodes:
                print(f"\n=== NODES ({len(results.nodes)}) ===")
                for node in results.nodes:
                    print(f"UUID: {node.uuid}")
                    print(f"Name: {node.name}")
                    print(f"Labels: {', '.join(node.labels)}")
                    if hasattr(node, 'summary') and node.summary:
                        summary = node.summary[:100] + '...' if len(node.summary) > 100 else node.summary
                        print(f"Summary: {summary}")
                    if hasattr(node, 'attributes') and node.attributes:
                        print("Attributes:")
                        for key, value in node.attributes.items():
                            print(f"  {key}: {value}")
                    print("---")
            
            # Display episodes
            if hasattr(results, "episodes") and results.episodes:
                print(f"\n=== EPISODES ({len(results.episodes)}) ===")
                for episode in results.episodes:
                    print(f"UUID: {episode.uuid}")
                    print(f"Name: {episode.name}")
                    if hasattr(episode, 'content'):
                        content = episode.content[:150] + '...' if len(episode.content) > 150 else episode.content
                        print(f"Content: {content}")
                    if hasattr(episode, 'source'):
                        print(f"Source: {episode.source}")
                    print("---")
            
            # Display communities
            if hasattr(results, "communities") and results.communities:
                print(f"\n=== COMMUNITIES ({len(results.communities)}) ===")
                for community in results.communities:
                    print(f"UUID: {community.uuid}")
                    print(f"Name: {community.name}")
                    if hasattr(community, 'summary') and community.summary:
                        summary = community.summary[:100] + '...' if len(community.summary) > 100 else community.summary
                        print(f"Summary: {summary}")
                    print("---")
        
        else:
            # Legacy display for list of results (typically edges)
            for result in results:
                print(f'UUID: {result.uuid}')
                print(f'Fact: {result.fact}')
                if hasattr(result, 'valid_at') and result.valid_at:
                    print(f'Valid from: {result.valid_at}')
                if hasattr(result, 'invalid_at') and result.invalid_at:
                    print(f'Valid until: {result.invalid_at}')
                print('---')
                for episode_uuid in result.episodes:
                    episode = await get_episode_by_uuid(graphiti, episode_uuid)
                    if episode:
                        print(f"Content: {episode['content']}")

            
    finally:
        #################################################
        # CLEANUP
        #################################################
        # Always close the connection to Neo4j when
        # finished to properly release resources
        #################################################

        # Close the connection
        await graphiti.close()
        print('\nConnection closed')


if __name__ == '__main__':  
    asyncio.run(main())
