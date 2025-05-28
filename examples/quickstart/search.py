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

LLM 제어 가능한 그래프 검색 시스템

주요 기능:
1. 텍스트 검색 (BM25 + Cosine): 구체적인 명사(회사명, 이름) 검색
2. 유사도 검색 (Cosine): 대략적인 지역/분야 검색  
3. 그래프 거리 검색 (BFS + Cosine): 특정 노드 중심 검색
4. 다단계 검색: 복잡한 쿼리를 여러 단계로 분해하여 검색
5. 스마트 검색: 쿼리 분석 후 자동 방법 선택

사용 예시:
- "삼성전자 직원" → 텍스트 검색으로 삼성전자 찾기 → 그래프 검색으로 직원 찾기
- "중동 지역 VC" → 유사도 검색으로 중동 국가 찾기 → 그래프 검색으로 VC 찾기
- "중동 지역의 정부 관련 부서 사람" → 다단계 검색 (국가→정부부서→사람)

서비스 계정 키 사용 예시:
1. JSON 파일 경로 사용:
   service_account_key_path="/path/to/service-account-key.json"

2. JSON 딕셔너리 직접 사용:
   service_account_key_json={
       "type": "service_account",
       "project_id": "your-project-id",
       "private_key_id": "...",
       "private_key": "...",
       "client_email": "...",
       "client_id": "...",
       "auth_uri": "...",
       "token_uri": "...",
       "auth_provider_x509_cert_url": "...",
       "client_x509_cert_url": "..."
   }

3. 환경 변수 사용:
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
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

# LLM and Embedding configuration
api_key = os.environ.get("LLM_API_KEY")
llm_model = os.environ.get("LLM_MODEL", "gemini-2.5-flash-preview-05-20")
embedding_model = os.environ.get("EMBEDDING_MODEL", "text-multilingual-embedding-002")
embedding_dim = int(os.environ.get("EMBEDDING_DIM", "768"))

# Vertex AI / Service Account configuration
service_account_key_path = os.environ.get("SERVICE_ACCOUNT_KEY_PATH")
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# 서비스 계정 키 JSON을 직접 로드하는 예시 (선택사항)
service_account_key_json = None
if service_account_key_path and os.path.exists(service_account_key_path):
    with open(service_account_key_path, 'r') as f:
        service_account_key_json = json.load(f)

if not api_key and not service_account_key_json and not service_account_key_path:
    raise ValueError("LLM_API_KEY 또는 서비스 계정 키가 설정되어야 합니다")

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


async def llm_controlled_search(
    graphiti,
    query: str,
    # 검색 방법 제어
    use_bm25: bool = True,
    use_cosine_similarity: bool = True, 
    use_bfs: bool = False,
    # 노드 필터
    node_labels: list = None,  # ["Person", "Company", "Country", "Industry"] 등
    # 엣지 검색 포함 여부
    include_edges: bool = False,
    # 결과 수 제한
    limit: int = 5,
    # 그래프 검색용 파라미터
    bfs_origin_node_uuids: list = None,
    bfs_max_depth: int = 2,
    # 기타 설정
    sim_min_score: float = 0.3,
    reranker_min_score: float = 0.1
):
    """
    LLM이 파라미터를 제어하여 동적으로 SearchConfig를 생성하고 검색을 수행하는 함수
    
    Args:
        graphiti: Graphiti 인스턴스
        query: 검색 쿼리
        use_bm25: BM25 텍스트 검색 사용 여부
        use_cosine_similarity: 코사인 유사도 검색 사용 여부
        use_bfs: BFS 그래프 검색 사용 여부
        node_labels: 필터링할 노드 라벨 리스트
        include_edges: 엣지 검색 포함 여부
        limit: 반환할 결과 수
        bfs_origin_node_uuids: BFS 시작 노드들의 UUID 리스트
        bfs_max_depth: BFS 최대 깊이
        sim_min_score: 유사도 최소 점수
        reranker_min_score: 재순위 최소 점수
        
    Returns:
        검색 결과
    """
    
    # 노드 검색 방법 설정
    node_search_methods = []
    if use_cosine_similarity:
        node_search_methods.append(NodeSearchMethod.cosine_similarity)
    if use_bm25:
        node_search_methods.append(NodeSearchMethod.bm25)
    if use_bfs:
        node_search_methods.append(NodeSearchMethod.bfs)
    
    # 최소 하나의 검색 방법은 있어야 함
    if not node_search_methods:
        node_search_methods = [NodeSearchMethod.cosine_similarity]
    
    # 노드 검색 설정
    node_config = NodeSearchConfig(
        search_methods=node_search_methods,
        reranker=NodeReranker.rrf,
        sim_min_score=sim_min_score,
        bfs_max_depth=bfs_max_depth
    )
    
    # 엣지 검색 설정 (필요시)
    edge_config = None
    if include_edges:
        edge_search_methods = []
        if use_cosine_similarity:
            edge_search_methods.append(EdgeSearchMethod.cosine_similarity)
        if use_bm25:
            edge_search_methods.append(EdgeSearchMethod.bm25)
        if use_bfs:
            edge_search_methods.append(EdgeSearchMethod.bfs)
        
        if edge_search_methods:
            edge_config = EdgeSearchConfig(
                search_methods=edge_search_methods,
                reranker=EdgeReranker.rrf,
                bfs_max_depth=bfs_max_depth
            )
    
    # SearchConfig 생성
    search_config = SearchConfig(
        node_config=node_config,
        edge_config=edge_config,
        limit=limit,
        reranker_min_score=reranker_min_score
    )
    
    # 검색 필터 설정
    search_filter = SearchFilters()
    if node_labels:
        search_filter.node_labels = node_labels
    
    # 검색 실행 - bfs_origin_node_uuids 파라미터 추가
    return await graphiti.search_(
        query,
        search_filter=search_filter,
        bfs_origin_node_uuids=bfs_origin_node_uuids,
        config=search_config
    )


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
    # LLM 설정 (서비스 계정 키 또는 API 키 사용)
    llm_config = LLMConfig(
        api_key=api_key,
        model=llm_model,
        service_account_key_path=service_account_key_path,
        service_account_key_json=service_account_key_json,
        project_id=project_id,
        location=location
    )
    
    # Embedder 설정 (서비스 계정 키 또는 API 키 사용)
    embedder_config = GeminiEmbedderConfig(
        api_key=api_key,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        service_account_key_path=service_account_key_path,
        service_account_key_json=service_account_key_json,
        project_id=project_id,
        location=location
    )
    
    graphiti = Graphiti(
        neo4j_uri,
        neo4j_user,
        neo4j_password,
        llm_client=GeminiClient(config=llm_config),
        embedder=GeminiEmbedder(config=embedder_config),
        cross_encoder=GeminiRerankerClient(config=llm_config)
    )


    try:
        # LLM 제어 가능한 검색 함수 사용 예시
        print("\n검색 방법을 선택하세요:")
        print("1. 텍스트 검색 (BM25 + Cosine)")
        print("2. 유사도 검색 (Cosine만)")
        print("3. 그래프 검색 (BFS + Cosine)")
        print("4. 포괄적 검색 (모든 방법 + 엣지)")
        print("5. 커스텀 검색")
        
        choice = input("선택 (1-5): ").strip()
        
        if choice == "1":
            # 텍스트 검색: 구체적인 명사 검색에 적합
            results = await llm_controlled_search(
                graphiti,
                search_query,
                use_bm25=True,
                use_cosine_similarity=True,
                use_bfs=False,
                node_labels=["Entity"],  # 또는 ["Person", "Company"] 등
                include_edges=False,
                limit=5
            )
            
        elif choice == "2":
            # 유사도 검색: 대략적인 지역/분야 검색에 적합
            results = await llm_controlled_search(
                graphiti,
                search_query,
                use_bm25=False,
                use_cosine_similarity=True,
                use_bfs=False,
                node_labels=["Entity"],
                include_edges=False,
                limit=5
            )
            
        elif choice == "3":
            # 그래프 검색: 특정 노드 중심 검색
            origin_uuids_input = input("BFS 시작 노드 UUID들 (쉼표로 구분, 선택사항): ").strip()
            bfs_origin_node_uuids = None
            if origin_uuids_input:
                bfs_origin_node_uuids = [uuid.strip() for uuid in origin_uuids_input.split(",") if uuid.strip()]
            
            bfs_depth_input = input("BFS 최대 깊이 (기본 2): ").strip()
            bfs_depth = int(bfs_depth_input) if bfs_depth_input.isdigit() else 2
            
            results = await llm_controlled_search(
                graphiti,
                search_query,
                use_bm25=False,
                use_cosine_similarity=True,
                use_bfs=True,
                node_labels=["Entity"],
                include_edges=False,
                bfs_origin_node_uuids=bfs_origin_node_uuids,
                bfs_max_depth=bfs_depth,
                limit=5
            )
            
        elif choice == "4":
            # 포괄적 검색: 모든 방법 + 엣지 포함
            results = await llm_controlled_search(
                graphiti,
                search_query,
                use_bm25=True,
                use_cosine_similarity=True,
                use_bfs=False,
                node_labels=None,  # 모든 노드 타입
                include_edges=True,
                limit=10
            )
            
        elif choice == "5":
            # 커스텀 검색: 사용자가 파라미터 직접 설정
            print("\n=== 커스텀 검색 설정 ===")
            use_bm25 = input("BM25 텍스트 검색 사용? (y/n): ").lower() == 'y'
            use_cosine = input("코사인 유사도 검색 사용? (y/n): ").lower() == 'y'
            use_bfs = input("BFS 그래프 검색 사용? (y/n): ").lower() == 'y'
            include_edges = input("엣지 검색 포함? (y/n): ").lower() == 'y'
            
            node_types = input("노드 타입 필터 (예: Person,Company 또는 엔터로 전체): ").strip()
            node_labels = [t.strip() for t in node_types.split(",")] if node_types else None
            
            limit = input("결과 수 제한 (기본 5): ").strip()
            limit = int(limit) if limit.isdigit() else 5
            
            bfs_origin_node_uuids = None
            bfs_max_depth = 2
            
            if use_bfs:
                origin_uuids_input = input("BFS 시작 노드 UUID들 (쉼표로 구분, 선택사항): ").strip()
                if origin_uuids_input:
                    # 쉼표로 분리하고 빈 문자열 제거
                    bfs_origin_node_uuids = [uuid.strip() for uuid in origin_uuids_input.split(",") if uuid.strip()]
                
                bfs_depth_input = input("BFS 최대 깊이 (기본 2): ").strip()
                bfs_max_depth = int(bfs_depth_input) if bfs_depth_input.isdigit() else 2
            
            results = await llm_controlled_search(
                graphiti,
                search_query,
                use_bm25=use_bm25,
                use_cosine_similarity=use_cosine,
                use_bfs=use_bfs,
                node_labels=node_labels,
                include_edges=include_edges,
                bfs_origin_node_uuids=bfs_origin_node_uuids,
                bfs_max_depth=bfs_max_depth,
                limit=limit
            )
            
        else:
            # 기본 검색
            results = await llm_controlled_search(
                graphiti,
                search_query,
                use_bm25=True,
                use_cosine_similarity=True
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
        
        # else:
        #     # Legacy display for list of results (typically edges)
        #     for result in results:
        #         print(f'UUID: {result.uuid}')
        #         print(f'Fact: {result.fact}')
        #         if hasattr(result, 'valid_at') and result.valid_at:
        #             print(f'Valid from: {result.valid_at}')
        #         if hasattr(result, 'invalid_at') and result.invalid_at:
        #             print(f'Valid until: {result.invalid_at}')
        #         print('---')
        #         for episode_uuid in result.episodes:
        #             episode = await get_episode_by_uuid(graphiti, episode_uuid)
        #             if episode:
        #                 print(f"Content: {episode['content']}")

            
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
    # 일반 검색 실행
    asyncio.run(main())
    
    # 예시 실행을 원하면 아래 주석 해제
    # asyncio.run(llm_search_examples())

