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
import pathlib
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv
from pyairtable import Api
from google.genai.errors import ClientError

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from pydantic import BaseModel, Field
from company_research import CompanyResearcher

class Person(BaseModel):
    """A Person entity in the system"""
    full_name: str | None = Field(..., description="The full name of the Person")
    email: str | None = Field(..., description="The email address of the Person")
    current_company: str | None = Field(..., description="The company where the Person currently works")
    location: str | None = Field(..., description="The geographic location of the Person")
    title: str | None = Field(..., description="The job title of the Person")

class Company(BaseModel):
    """A company or business entity"""
    company_name: str | None = Field(..., description="The name of the company")
    industry: str | None = Field(..., description="The industry the company operates in")
    country: str | None = Field(..., description="The country where the company is headquartered")
    number_of_employee: int | None = Field(..., description="The number of employees in the company")

class Industry(BaseModel):
    """Industry classification based on GICS standards"""
    industry_name: str | None = Field(..., description="Name of the industry sector")

class Country(BaseModel):
    """Country information"""
    country_name: str | None = Field(..., description="Official name of the country")
    region: str | None = Field(..., description="Region where the country is located (Asia, Europe, North America, etc.)")

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
api_key = os.environ.get("LLM_API_KEY")
llm_model = os.environ.get("LLM_MODEL", "gemini-2.5-flash-preview-05-20")
embedding_model = os.environ.get("EMBEDDING_MODEL", "text-multilingual-embedding-002")
embedding_dim = int(os.environ.get("EMBEDDING_DIM", "768"))

# Neo4j connection parameters
# Make sure Neo4j Desktop is running with a local DBMS started
neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')

# Vertex AI / Service Account configuration
service_account_key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# 서비스 계정 키 JSON을 직접 로드하는 예시 (선택사항)
service_account_key_json = None
if service_account_key_path and os.path.exists(service_account_key_path):
    with open(service_account_key_path, 'r') as f:
        service_account_key_json = json.load(f)

# Airtable connection parameters
airtable_api_key = os.environ.get('AIRTABLE_API_KEY')
airtable_base_id = os.environ.get('AIRTABLE_BASE_ID')
contacts_table_id = os.environ.get('AIRTABLE_CONTACTS_TABLE_ID', 'Contacts')

if not api_key and not service_account_key_json and not service_account_key_path:
    raise ValueError("LLM_API_KEY 또는 서비스 계정 키가 설정되어야 합니다")

if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

if not airtable_api_key or not airtable_base_id:
    raise ValueError('AIRTABLE_API_KEY and AIRTABLE_BASE_ID must be set')


async def fetch_airtable_contacts():
    """Fetch contact data from Airtable Contacts table and save to file"""
    logger.info("Checking for cached contacts data...")
    
    # 캐시 파일 경로 설정
    cache_dir = pathlib.Path("cache")
    cache_dir.mkdir(exist_ok=True)
    contacts_cache_file = cache_dir / "contacts_cache.json"
    
    # 캐시된 데이터가 있으면 불러오기
    if contacts_cache_file.exists():
        try:
            with open(contacts_cache_file, 'r', encoding='utf-8') as f:
                contacts = json.load(f)
            logger.info(f"Loaded {len(contacts)} contacts from cache")
            return contacts
        except Exception as e:
            logger.error(f"Error loading contacts cache file: {e}")
    
    # 캐시된 데이터가 없으면 에어테이블에서 가져오기
    logger.info("Fetching contacts data from Airtable...")
    
    api = Api(airtable_api_key)
    contacts_table = api.table(airtable_base_id, contacts_table_id)
    contacts = contacts_table.all()
    
    # 가져온 데이터를 캐시 파일에 저장
    try:
        with open(contacts_cache_file, 'w', encoding='utf-8') as f:
            json.dump(contacts, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(contacts)} contacts to cache")
    except Exception as e:
        logger.error(f"Error saving contacts to cache file: {e}")
    
    logger.info(f"Fetched {len(contacts)} contacts from Airtable")
    return contacts


async def import_airtable_data_to_graphiti(graphiti):
    """Import Airtable contacts into Graphiti using CompanyResearcher for company data"""
    
    contacts = await fetch_airtable_contacts()
    
    # Define entity types dictionary to pass to add_episode
    entity_types = {"Person": Person, "Company": Company, "Industry": Industry, "Country": Country}
    
    # Initialize company researcher for Gemini API
    company_researcher = CompanyResearcher(api_key)
    
    # Create cache directory for company data
    cache_dir = pathlib.Path("cache")
    cache_dir.mkdir(exist_ok=True)
    companies_cache_file = cache_dir / "city_based_companies_cache.json"
    
    # Load existing company cache if available
    companies_cache = {}
    if companies_cache_file.exists():
        try:
            with open(companies_cache_file, 'r', encoding='utf-8') as f:
                companies_cache = json.load(f)
            logger.info(f"Loaded {len(companies_cache)} companies from cache")
        except Exception as e:
            logger.error(f"Error loading cache file: {e}")
    
    # Track consecutive API errors
    consecutive_errors = 0
    
    # Import contacts data and research their associated companies
    for i, contact in enumerate(contacts[66:], start=66):
        fields = contact['fields']
        company_name = fields.get('Company Name', '').strip() if fields.get('Company Name') else None
        
        # Create contact episode text
        contact_text = ""
        
        # 기본 인적 정보 (이름, 회사, 직책)
        name_parts = []
        if fields.get('Full Name'):
            name_parts.append(fields.get('Full Name'))
        
        if name_parts:
            contact_text = f"{' '.join(name_parts)}"
            
            # 회사 정보 추가
            if company_name:
                contact_text += f" works at {company_name}"
                
                # 직책 정보 추가
                if fields.get('Title'):
                    contact_text += f" as a {fields.get('Title')}"
                    
            contact_text += ". "
        
        # 이메일 정보 추가
        if fields.get('Email'):
            contact_text += f"Their email address is {fields.get('Email')}. "
            
        # 부서 정보 추가
        if fields.get('Department'):
            contact_text += f"They work in the {fields.get('Department')} department. "
            
        # 위치 정보 추가
        if fields.get('Country/Region'):
            contact_text += f"They are located in {fields.get('Country/Region')}."
        
        # Initialize company text
        company_text = ""
        
        # Research company data if company name is available
        if company_name:
            # Extract location context from contact data
            location_context = None
            location_parts = []
            
            # Add city information if available
            if fields.get('City'):
                location_parts.append(fields.get('City'))
            
            # Add country/region information if available
            if fields.get('Country/Region'):
                location_parts.append(fields.get('Country/Region'))
            
            # Create location context string
            if location_parts:
                location_context = ', '.join(location_parts)
                logger.info(f"Using location context for {company_name}: {location_context}")
            
            # Check cache first
            cache_key = f"{company_name}_{location_context}" if location_context else company_name
            if cache_key in companies_cache:
                logger.info(f"Using cached data for company '{company_name}' with location context '{location_context}'")
                company_data = companies_cache[cache_key]
                company_text = f"{company_name}"
                
                # 산업 정보 추가
                if company_data.get('Industry') and company_data.get('Industry') != 'Unknown':
                    company_text += f" is a company in the {company_data.get('Industry')} industry"
                else:
                    company_text += " is a company"
                
                company_text += ". "
                
                # 본사 위치 정보 추가
                if company_data.get('Country/Region') and company_data.get('Country/Region') != 'Unknown':
                    company_text += f"It is headquartered in {company_data.get('Country/Region')}. "
                
                # 직원 수 정보 추가
                if company_data.get('Number_of_employee') or company_data.get('Number of employee'):
                    employees = company_data.get('Number_of_employee', company_data.get('Number of employee'))
                    if employees != 'Unknown':
                        company_text += f"The company has approximately {employees} employees. "
                
                # 회사 설명 추가
                if company_data.get('Description'):
                    company_text += f"{company_data.get('Description')}"
            else:
                # Research company using Gemini API
                logger.info(f"Researching company '{company_name}' with Gemini API...")
                try:
                    researched_company = await company_researcher.research_company(company_name, location_context)
                    if researched_company:
                        # Update cache with new company data
                        companies_cache[cache_key] = researched_company
                        
                        # Save updated cache
                        try:
                            with open(companies_cache_file, 'w', encoding='utf-8') as f:
                                json.dump(companies_cache, f, ensure_ascii=False, indent=2)
                            logger.info(f"Added researched company '{company_name}' to cache")
                        except Exception as e:
                            logger.error(f"Error updating cache file: {e}")
                        
                        # Create company text from researched data
                        company_text = f"{company_name}"
                        
                        # 산업 정보 추가
                        if researched_company.get('Industry') and researched_company.get('Industry') != 'Unknown':
                            company_text += f" is a company in the {researched_company.get('Industry')} industry"
                        else:
                            company_text += " is a company"
                        
                        company_text += ". "
                        
                        # 본사 위치 정보 추가
                        if researched_company.get('Country/Region') and researched_company.get('Country/Region') != 'Unknown':
                            company_text += f"It is headquartered in {researched_company.get('Country/Region')}. "
                        
                        # 직원 수 정보 추가
                        if researched_company.get('Number_of_employee') or researched_company.get('Number of employee'):
                            employees = researched_company.get('Number_of_employee', researched_company.get('Number of employee'))
                            if employees != 'Unknown':
                                company_text += f"The company has approximately {employees} employees. "
                        
                        # 회사 설명 추가
                        if researched_company.get('Description'):
                            company_text += f"{researched_company.get('Description')}"
                except ClientError as e:
                    # Handle API rate limits with exponential backoff
                    error_message = str(e).lower()
                    if '429' in error_message or 'resource_exhausted' in error_message or 'rate limit' in error_message:
                        consecutive_errors += 1
                        wait_time = min(30 * (2 ** (consecutive_errors - 1)), 300)  # Max 5 minutes
                        logger.warning(f"API rate limit reached. Waiting for {wait_time} seconds. Error: {str(e)}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Error researching company '{company_name}': {str(e)}")
                except Exception as e:
                    logger.error(f"Error researching company '{company_name}': {str(e)}")
        
        # Combine contact and company information into one episode
        episode_content = contact_text
        if company_text:
            episode_content += f"\n\n{company_text}"
        
        # API 요청 제한 처리를 위한 재시도 로직
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Add episode for contact with company information
                await graphiti.add_episode(
                    name=f"Contact: {fields.get('Full Name', 'Unknown')} at {company_name or 'Unknown Company'}",
                    episode_body=episode_content,
                    source=EpisodeType.text,
                    source_description='Contact and Company Information',
                    reference_time=datetime.now(timezone.utc),
                    entity_types=entity_types
                )
                logger.info(f"Added contact {i+1}/{len(contacts)}: {fields.get('Full Name', 'Unknown')} at {company_name or 'Unknown Company'}")
                
                # Reset consecutive errors counter on success
                consecutive_errors = 0
                break  # 성공하면 재시도 루프 종료
                
            except ClientError as e:
                # 429 RESOURCE_EXHAUSTED 또는 rate limit 관련 메시지인 경우에만 재시도
                error_message = str(e).lower()
                if '429' in error_message or 'resource_exhausted' in error_message or 'rate limit' in error_message or 'quota' in error_message:
                    retry_count += 1
                    consecutive_errors += 1
                    wait_time = 30 * (2 ** (consecutive_errors - 1))  # 지수 백오프: 30초, 60초, 120초...
                    if wait_time > 300:  # 최대 5분까지만 대기
                        wait_time = 300
                    
                    logger.warning(f"API rate limit reached. Waiting for {wait_time} seconds before retry {retry_count}/{max_retries}. Error: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    # 다른 종류의 ClientError는 일반 오류로 처리
                    logger.error(f"Client error adding episode: {str(e)}")
                    break
                    
            except Exception as e:
                logger.error(f"Error adding contact episode: {str(e)}")
                break  # 다른 종류의 오류는 재시도하지 않음
        
        if retry_count == max_retries:
            logger.error(f"Failed to add contact after {max_retries} retries: {fields.get('Full Name', 'Unknown')}")
    
    logger.info("Completed importing Airtable data to Graphiti")


async def main():
    #################################################
    # INITIALIZATION
    #################################################
    # Connect to Neo4j and set up Graphiti indices
    # This is required before using other Graphiti
    # functionality
    #################################################

    # Initialize Graphiti with Neo4j connection
    #graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)

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
        # Initialize the graph database with graphiti's indices. This only needs to be done once.
        await graphiti.build_indices_and_constraints()

        #################################################
        # AIRTABLE DATA IMPORT
        #################################################
        # Import contacts from Airtable and research
        # company information using CompanyResearcher
        #################################################
        
        # Import Airtable data
        await import_airtable_data_to_graphiti(graphiti)

        #################################################
        # ADDING EPISODES
        #################################################
        # Episodes are the primary units of information
        # in Graphiti. They can be text or structured JSON
        # and are automatically processed to extract entities
        # and relationships.
        #################################################

        # Example: Add Episodes
        # Episodes list containing both text and JSON episodes

        #################################################
        # BASIC SEARCH
        #################################################
        # The simplest way to retrieve relationships (edges)
        # from Graphiti is using the search method, which
        # performs a hybrid search combining semantic
        # similarity and BM25 text retrieval.
        #################################################

        # Perform a hybrid search combining semantic similarity and BM25 retrieval
        print("\nSearching for: 'Who works in samsung?'")
        results = await graphiti.search('Who works in samsung?')

        # Print search results
        print('\nSearch Results:')
        for result in results:
            print(f'UUID: {result.uuid}')
            print(f'Fact: {result.fact}')
            if hasattr(result, 'valid_at') and result.valid_at:
                print(f'Valid from: {result.valid_at}')
            if hasattr(result, 'invalid_at') and result.invalid_at:
                print(f'Valid until: {result.invalid_at}')
            print('---')

        #################################################
        # CENTER NODE SEARCH
        #################################################
        # For more contextually relevant results, you can
        # use a center node to rerank search results based
        # on their graph distance to a specific node
        #################################################

        # Use the top search result's UUID as the center node for reranking
        # if results and len(results) > 0:
        #     # Get the source node UUID from the top result
        #     center_node_uuid = results[0].source_node_uuid

        #     print('\nReranking search results based on graph distance:')
        #     print(f'Using center node UUID: {center_node_uuid}')

        #     reranked_results = await graphiti.search(
        #         'Who was the California Attorney General?', center_node_uuid=center_node_uuid
        #     )

        #     # Print reranked search results
        #     print('\nReranked Search Results:')
        #     for result in reranked_results:
        #         print(f'UUID: {result.uuid}')
        #         print(f'Fact: {result.fact}')
        #         if hasattr(result, 'valid_at') and result.valid_at:
        #             print(f'Valid from: {result.valid_at}')
        #         if hasattr(result, 'invalid_at') and result.invalid_at:
        #             print(f'Valid until: {result.invalid_at}')
        #         print('---')
        # else:
        #     print('No results found in the initial search to use as center node.')

        # #################################################
        # # NODE SEARCH USING SEARCH RECIPES
        # #################################################
        # # Graphiti provides predefined search recipes
        # # optimized for different search scenarios.
        # # Here we use NODE_HYBRID_SEARCH_RRF for retrieving
        # # nodes directly instead of edges.
        # #################################################

        # # Example: Perform a node search using _search method with standard recipes
        # print(
        #     '\nPerforming node search using _search method with standard recipe NODE_HYBRID_SEARCH_RRF:'
        # )

        # # Use a predefined search configuration recipe and modify its limit
        # node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        # node_search_config.limit = 5  # Limit to 5 results

        # # Execute the node search
        # node_search_results = await graphiti._search(
        #     query='California Governor',
        #     config=node_search_config,
        # )

        # # Print node search results
        # print('\nNode Search Results:')
        # for node in node_search_results.nodes:
        #     print(f'Node UUID: {node.uuid}')
        #     print(f'Node Name: {node.name}')
        #     node_summary = node.summary[:100] + '...' if len(node.summary) > 100 else node.summary
        #     print(f'Content Summary: {node_summary}')
        #     print(f'Node Labels: {", ".join(node.labels)}')
        #     print(f'Created At: {node.created_at}')
        #     if hasattr(node, 'attributes') and node.attributes:
        #         print('Attributes:')
        #         for key, value in node.attributes.items():
        #             print(f'  {key}: {value}')
        #     print('---')

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
