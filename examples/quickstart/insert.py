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
import time
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv
import requests
from pyairtable import Api
from google.genai.errors import ClientError

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.openai_client import OpenAIClient, LLMConfig
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF   
from pydantic import BaseModel, Field

from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from company_research import CompanyResearcher

class Contact(BaseModel):
    """A contact entity in the system"""
    contact_name: str | None = Field(..., description="The full name of the contact")
    email: str | None = Field(..., description="The email address of the contact")
    current_company: str | None = Field(..., description="The company where the contact currently works")
    department: str | None = Field(..., description="The department the contact works in")
    #phone: str | None = Field(..., description="The phone number of the contact")
    #linkedin_url: str | None = Field(..., description="The LinkedIn profile URL of the contact")
    location: str | None = Field(..., description="The geographic location of the contact")
    title: str | None = Field(..., description="The job title of the contact")

class Company(BaseModel):
    """A company or business entity"""
    company_name: str | None = Field(..., description="The name of the company")
    industry: str | None = Field(..., description="The industry the company operates in")
    country: str | None = Field(..., description="The country where the company is headquartered")
    number_of_employee: int | None = Field(..., description="The number of employees in the company")
    description: str | None = Field(..., description="Brief description of the company")    


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

# Neo4j connection parameters
# Make sure Neo4j Desktop is running with a local DBMS started
neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')

# Airtable connection parameters
airtable_api_key = os.environ.get('AIRTABLE_API_KEY')
airtable_base_id = os.environ.get('AIRTABLE_BASE_ID')
contacts_table_id = os.environ.get('AIRTABLE_CONTACTS_TABLE_ID', 'Contacts')
companies_table_id = os.environ.get('AIRTABLE_COMPANIES_TABLE_ID', 'Companies')

if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

if not airtable_api_key or not airtable_base_id:
    raise ValueError('AIRTABLE_API_KEY and AIRTABLE_BASE_ID must be set')


async def fetch_airtable_data():
    """Fetch data from Airtable for Contacts and Companies tables (limited to 200 records each)"""
    logger.info("Fetching data from Airtable...")
    
    api = Api(airtable_api_key)
    
    # Fetch contacts (limited to 200 records)
    contacts_table = api.table(airtable_base_id, contacts_table_id)
    contacts = contacts_table.all(max_records=300)
    
    # 회사 데이터를 저장할 파일 경로
    cache_dir = pathlib.Path("cache")
    # 캐시 디렉토리가 없으면 생성
    cache_dir.mkdir(exist_ok=True)
    companies_cache_file = cache_dir / "companies_cache.json"
    companies_dict = {}
    
    # 캐시 파일이 있는지 확인
    if companies_cache_file.exists():
        logger.info("Loading companies data from cache...")
        try:
            with open(companies_cache_file, 'r', encoding='utf-8') as f:
                companies_dict = json.load(f)
            logger.info(f"Loaded {len(companies_dict)} companies from cache")
        except Exception as e:
            logger.error(f"Error loading cache file: {e}")
            companies_dict = {}
    
    # 캐시가 없거나 비어 있으면 에어테이블에서 가져옴
    if not companies_dict:
        logger.info("Fetching companies data from Airtable...")
        # Fetch companies
        companies_table = api.table(airtable_base_id, companies_table_id)
        companies = companies_table.all()
        
        # Create a dictionary of companies for easier lookup
        for company in companies:
            if 'Company Name' in company['fields']:
                company_name = company['fields']['Company Name'].strip()
                companies_dict[company_name] = company['fields']
        
        # 캐시 파일에 저장
        try:
            with open(companies_cache_file, 'w', encoding='utf-8') as f:
                json.dump(companies_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"Cached {len(companies_dict)} companies to {companies_cache_file}")
        except Exception as e:
            logger.error(f"Error writing cache file: {e}")
    
    logger.info(f"Fetched {len(contacts)} contacts and {len(companies_dict)} companies")
    
    return contacts, companies_dict


async def import_airtable_data_to_graphiti(graphiti):
    """Import Airtable data into Graphiti using custom entity types as JSON episodes"""
    
    contacts, companies_dict = await fetch_airtable_data()
    
    # Define entity types dictionary to pass to add_episode
    entity_types = {"Contact": Contact, "Company": Company}
    
    # Initialize company researcher for Gemini API
    api_key = os.environ.get("LLM_API_KEY")
    company_researcher = CompanyResearcher(api_key)
    
    # 연속 API 오류 추적용 변수
    consecutive_errors = 0
    
    # Import contacts data with their associated company information
    for contact in contacts:
        fields = contact['fields']
        company_name = fields.get('Company Name', '').strip() if fields.get('Company Name') else None
        
        # Create structured contact data
        contact_data = {
            "contact_name": fields.get('Full Name'),
            "email": fields.get('Email'),
            "current_company": company_name,
            "department": fields.get('Department'),
            "location": fields.get('Country/Region'),
            "title": fields.get('Title')
        }
        
        # Initialize company data
        company_data = None
        
        # 모든 회사에 대해 제미니 API로 정보 조사
        if company_name:
            # 캐시 파일 경로
            cache_dir = pathlib.Path("cache")
            companies_cache_file = cache_dir / "companies_cache.json"
            
            # 캐시에서 먼저 확인
            cached_data = None
            if companies_cache_file.exists():
                try:
                    with open(companies_cache_file, 'r', encoding='utf-8') as f:
                        cache = json.load(f)
                    cached_data = cache.get(company_name)
                except Exception as e:
                    logger.error(f"Error reading cache: {e}")
            
            # 캐시에 있으면 사용, 없으면 제미니 API 호출
            company_fields = None
            if cached_data:
                logger.info(f"Using cached data for company '{company_name}'")
                company_fields = cached_data
            else:
                # 모든 회사에 대해 제미니 API 호출
                logger.info(f"Researching company '{company_name}' with Gemini API...")
                try:
                    researched_company = await company_researcher.research_company(company_name)
                    if researched_company:
                        company_fields = researched_company
                        
                        # 캐시 업데이트
                        try:
                            # 기존 캐시 로드
                            current_cache = {}
                            if companies_cache_file.exists():
                                with open(companies_cache_file, 'r', encoding='utf-8') as f:
                                    current_cache = json.load(f)
                            
                            # 새 회사 데이터 추가
                            current_cache[company_name] = researched_company
                            
                            # 캐시 저장
                            with open(companies_cache_file, 'w', encoding='utf-8') as f:
                                json.dump(current_cache, f, ensure_ascii=False, indent=2)
                            logger.info(f"Added researched company '{company_name}' to cache")
                        except Exception as e:
                            logger.error(f"Error updating cache file: {e}")
                except Exception as e:
                    logger.error(f"Error researching company '{company_name}': {e}")
            
            # Create structured company data if company information is available
            if company_fields:
                company_data = {
                    "company_name": company_name,
                    "industry": company_fields.get('Industry'),
                    "country": company_fields.get('Country/Region'),
                    "number_of_employee": company_fields.get('Number_of_employee', company_fields.get('Number of employee')),
                    "description": company_fields.get('Description')
                }
        
        # Combine contact and company data into a single JSON structure
        episode_data = {
            "contact": contact_data
        }
        
        if company_data:
            episode_data["company"] = company_data
        
        # Convert dictionary to JSON string
        episode_json_str = json.dumps(episode_data, ensure_ascii=False)
        
        # API 요청 제한 처리를 위한 재시도 로직
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Add episode for each contact with company information using JSON format
                await graphiti.add_episode(
                    name=f"Contact: {fields.get('Full Name', 'Unknown')} at {company_name or 'Unknown Company'}",
                    episode_body=episode_json_str,  # Pass JSON string instead of dictionary
                    source=EpisodeType.json,
                    source_description='Contact and Company Data',
                    reference_time=datetime.now(timezone.utc),
                    entity_types=entity_types,
                )
                logger.info(f"Added contact with company info: {fields.get('Full Name', 'Unknown')} at {company_name or 'Unknown Company'}")
                consecutive_errors = 0  # 성공 시 연속 오류 카운트 초기화
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
                logger.error(f"Error adding episode: {str(e)}")
                break  # 다른 종류의 오류는 재시도하지 않음
        
        if retry_count == max_retries:
            logger.error(f"Failed to add contact after {max_retries} retries: {fields.get('Full Name', 'Unknown')}")
    
    logger.info("Completed importing Airtable data to Graphiti")


async def add_test_contact_to_graphiti(graphiti, contact_info):
    # Define entity types dictionary to pass to add_episode
    entity_types = {"Contact": Contact, "Company": Company}
    
    # Extract contact information
    full_name = contact_info.get('full_name', 'Test Contact')
    company_name = contact_info.get('company_name', 'Test Company')
    
    # Create structured contact data
    contact_data = {
        "contact_name": full_name,
        "email": contact_info.get('email'),
        "current_company": company_name,
        "department": contact_info.get('department'),
        "location": contact_info.get('country'),
        "title": contact_info.get('title')
    }
    
    # Create structured company data if company information is provided
    company_data = None
    if any(key.startswith('company_') for key in contact_info.keys()):
        company_data = {
            "company_name": company_name,
            "industry": contact_info.get('company_industry'),
            "country": contact_info.get('company_country'),
            "number_of_employee": contact_info.get('company_employees'),
            "description": contact_info.get('company_description')
        }
    
    # Combine contact and company data
    episode_data = {
        "contact": contact_data
    }
    
    if company_data:
        episode_data["company"] = company_data
    
    try:
        # Convert dictionary to JSON string
        episode_json_str = json.dumps(episode_data, ensure_ascii=False)
        
        # Add episode for the test contact using JSON format
        await graphiti.add_episode(
            name=f"Contact: {full_name} at {company_name}",
            episode_body=episode_json_str,  # Pass JSON string
            source=EpisodeType.json,
            source_description='Test Contact Data',
            reference_time=datetime.now(timezone.utc),
            entity_types=entity_types,
        )
        logger.info(f"Added test contact: {full_name} at {company_name}")
        return True
    except Exception as e:
        logger.error(f"Error adding test contact: {str(e)}")
        return False


async def main():
    #################################################
    # INITIALIZATION
    #################################################
    # Connect to Neo4j and set up Graphiti indices
    # This is required before using other Graphiti
    # functionality
    #################################################

    # Initialize Graphiti with Neo4j connection
    
    """graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password, llm_client=OpenAIClient(
        config=LLMConfig(
            api_key=api_key,
            model="gpt-4o-mini"
        )
    ))
    """
    
    graphiti = Graphiti(
    neo4j_uri,
    neo4j_user,
    neo4j_password,
    llm_client=GeminiClient(
        config=LLMConfig(
            api_key=api_key,
            #model="gemini-2.0-flash"
            model="gemini-2.5-flash-preview-04-17"
            #model="gemini-2.5-pro-preview-05-06"
        )
    ),  
    embedder=GeminiEmbedder(
        config=GeminiEmbedderConfig(
            api_key=api_key,
            embedding_model="gemini-embedding-exp-03-07"
            )
        )
    )
    
    try:
        # Initialize the graph database with graphiti's indices. This only needs to be done once.
        await graphiti.build_indices_and_constraints()

        #################################################
        # IMPORT AIRTABLE DATA
        #################################################
        # Import contacts and companies from Airtable
        # and add them to Graphiti as episodes with
        # custom entity types
        #################################################
        
        # Example usage of test contact method
        test_contact = {
            'full_name': '김현구',
            'email': 'hyungu@sprintsolo.dev',
            'company_name': 'SprintSolo',
            'title': '대표',
            'department': '',
            'country': '대한민국',
            'company_industry': '소프트웨어 개발',
            'company_country': '대한민국',
            'company_employees': '3',
            'company_description': '스타트업을 위한 AI 개발 프로토타이핑 서비스.'
        }
        
        # Add test contact to Graphiti
        await add_test_contact_to_graphiti(graphiti, test_contact)

        await import_airtable_data_to_graphiti(graphiti)
        
        results = await graphiti.search('search_query')

        # Print search results
        print('\nSearch Results:')
        for result in results:
            print(f'UUID: {result.uuid}')
            print(f'Fact: {result.fact}')
            print(result.episodes)
            if hasattr(result, 'valid_at') and result.valid_at:
                print(f'Valid from: {result.valid_at}')
            if hasattr(result, 'invalid_at') and result.invalid_at:
                print(f'Valid until: {result.invalid_at}')
            print('---')


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
