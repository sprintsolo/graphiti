"""
Company research module using Gemini API.

This module provides functionality to research company information
when it's not found in the Airtable dictionary using Google's Gemini API.
"""

import os
import json
import logging
from typing import Dict, Optional, Any

from google import genai
from google.genai import types
from google.genai.types import Tool, GoogleSearch
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Schema for company information
class CompanyInfo(BaseModel):
    """Schema for company information"""
    Industry: str = Field(description="The industry the company operates in, using GICS Industry Group classification")
    Country_Region: str = Field(description="The country/region where the company is headquartered")
    Number_of_employee: int | str = Field(description="The approximate number of employees (just the number)")
    Description: str = Field(description="A brief one-sentence description of the company")

class CompanyResearcher:
    """Class to research company information using Gemini API with Google Search grounding."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-preview-04-17"):
        """
        Initialize the CompanyResearcher with the Gemini API.
        
        Args:
            api_key: Gemini API key
            model: Gemini model to use for research
        """
        self.api_key = api_key
        self.model = model
        self.client = genai.Client(api_key=api_key)
        
        # Create Google Search tool for grounding
        self.google_search_tool = Tool(
            google_search=GoogleSearch()
        )
        
        self.max_tokens = 1024
    
    async def research_company(self, company_name: str, location_context: str = None) -> Optional[Dict[str, Any]]:
        """
        Research company information using Gemini API with a two-step approach:
        1. First query with Google Search grounding to get information
        2. Second query without grounding to get structured output
        
        Args:
            company_name: Name of the company to research
            location_context: Optional location context (city/country) to help with company identification
            
        Returns:
            Dictionary with company information or None if unsuccessful
        """
        try:
            logger.info(f"Researching company: {company_name}")
            if location_context:
                logger.info(f"Using location context: {location_context}")
            
            # Step 1: Get information with Google Search grounding
            search_info = await self._research_with_search(company_name, location_context)
            if not search_info:
                logger.error(f"Failed to get information with search for {company_name}")
                return None
                
            # Step 2: Get structured output using the search results
            structured_data = await self._get_structured_data(company_name, search_info)
            if structured_data:
                return structured_data
                
            # Fallback: If structured output fails, try to parse the search info directly
            logger.info(f"Falling back to search info for {company_name}")
            try:
                # If search_info is already a dictionary, use it directly
                if isinstance(search_info, dict):
                    return search_info
                    
                # Try to extract JSON from the text
                if isinstance(search_info, str):
                    # Find JSON in the text if it exists
                    start_idx = search_info.find("{")
                    end_idx = search_info.rfind("}")
                    if start_idx != -1 and end_idx != -1:
                        json_str = search_info[start_idx:end_idx+1]
                        data = json.loads(json_str)
                        
                        # Fix key names if needed
                        if "Country_Region" in data:
                            data["Country/Region"] = data.pop("Country_Region")
                            
                        return data
            except Exception as e:
                logger.error(f"Error parsing fallback data: {e}")
                
            return None
            
        except Exception as e:
            logger.error(f"Error researching company {company_name}: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            return None
    
    async def _research_with_search(self, company_name: str, location_context: str = None) -> Optional[str]:
        """
        First step: Research company with Google Search grounding.
        
        Args:
            company_name: Name of the company to research
            location_context: Optional location context (city/country) to help with company identification
            
        Returns:
            Information about the company or None if unsuccessful
        """
        try:
            logger.info(f"Step 1: Researching {company_name} with Google Search")
            
            # Create the search query with location context if available
            search_query = f"Research and provide detailed information about the company: {company_name}"
            if location_context:
                search_query += f" (located in or related to {location_context})"
            
            # Create the prompt as a Content object
            content = types.Content(
                role="user",
                parts=[types.Part.from_text(
                    text=f"""{search_query}
                    
                    Focus on finding these specific details:
                    1. Industry the company operates in (please use GICS Industry Group classification) - IMPORTANT: Always use the English industry name even when responding in other languages
                    2. Country/Region where the company is headquartered
                    3. Number of employees (approximate)
                    4. A brief one-sentence description of what the company does (make it as concise as possible)
                    
                    {f"Additional context: The company may be related to or located in {location_context}. Please use this information to help identify the correct company if there are multiple companies with similar names." if location_context else ""}
                    
                    For the industry classification, specifically select ONE from this exact list of GICS Industry Groups:
                    1. Energy
                    2. Materials
                    3. Capital Goods
                    4. Commercial & Professional Services
                    5. Transportation
                    6. Automobiles & Components
                    7. Consumer Durables & Apparel
                    8. Consumer Services
                    9. Media & Entertainment
                    10. Retailing
                    11. Food & Staples Retailing
                    12. Food, Beverage & Tobacco
                    13. Household & Personal Products
                    14. Health Care Equipment & Services
                    15. Pharmaceuticals, Biotechnology & Life Sciences
                    16. Banks
                    17. Diversified Financials
                    18. Insurance
                    19. Real Estate
                    20. Software & Services
                    21. Technology Hardware & Equipment
                    22. Semiconductors & Semiconductor Equipment
                    23. Telecommunication Services
                    24. Utilities
                    
                    Provide comprehensive information so I can understand the company well.
                    """
                )]
            )
            
            # Generate content with grounding search
            generation_config = types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=self.max_tokens,
                tools=[self.google_search_tool],
            )
            
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=[content],
                config=generation_config,
            )
            
            # Get text response
            if response.candidates and response.candidates[0].content.parts:
                search_result = response.candidates[0].content.parts[0].text
                
                # Log grounding metadata if available
                if hasattr(response.candidates[0], 'grounding_metadata') and \
                   hasattr(response.candidates[0].grounding_metadata, 'search_entry_point'):
                    logger.info("Search grounding metadata available for step 1")
                
                logger.info(f"Successfully completed step 1 for {company_name}")
                return search_result
            else:
                logger.error("Empty response from search query")
                print(response)
                return None
                
        except Exception as e:
            logger.error(f"Error in search step: {str(e)}")
            return None
    
    async def _get_structured_data(self, company_name: str, search_info: str) -> Optional[Dict[str, Any]]:
        """
        Second step: Get structured data using the information from search.
        
        Args:
            company_name: Name of the company
            search_info: Information obtained from search
            
        Returns:
            Structured company data or None if unsuccessful
        """
        try:
            logger.info(f"Step 2: Getting structured data for {company_name}")
            
            # Create the prompt with search info
            content = types.Content(
                role="user",
                parts=[types.Part.from_text(
                    text=f"""Here's information about {company_name} that I've gathered:
                    
                    {search_info}
                    
                    Based on this information, provide the company details in a structured format.
                    
                    Make sure the Industry field uses EXACTLY one of these GICS Industry Group classifications:
                    1. Energy
                    2. Materials
                    3. Capital Goods
                    4. Commercial & Professional Services
                    5. Transportation
                    6. Automobiles & Components
                    7. Consumer Durables & Apparel
                    8. Consumer Services
                    9. Media & Entertainment
                    10. Retailing
                    11. Food & Staples Retailing
                    12. Food, Beverage & Tobacco
                    13. Household & Personal Products
                    14. Health Care Equipment & Services
                    15. Pharmaceuticals, Biotechnology & Life Sciences
                    16. Banks
                    17. Diversified Financials
                    18. Insurance
                    19. Real Estate
                    20. Software & Services
                    21. Technology Hardware & Equipment
                    22. Semiconductors & Semiconductor Equipment
                    23. Telecommunication Services
                    24. Utilities
                    """
                )]
            )
            
            # System instruction
            system_prompt = "Extract the information and return it in the same language as provided."
            
            # Generate content for structured output
            generation_config = types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json",
                response_schema=CompanyInfo,
                system_instruction=system_prompt,
            )
            
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=[content],
                config=generation_config,
            )
            
            try:
                # Parse response into Pydantic model
                validated_model = CompanyInfo.model_validate(json.loads(response.text))
                
                # Convert to dictionary
                company_dict = validated_model.model_dump()
                
                # Fix key names to match expected format in the main application
                if "Country_Region" in company_dict:
                    company_dict["Country/Region"] = company_dict.pop("Country_Region")
                
                logger.info(f"Successfully completed step 2 for {company_name}")
                return company_dict
                
            except Exception as e:
                logger.error(f"Error parsing structured response: {e}")
                # Try to extract JSON directly
                if hasattr(response, 'text'):
                    try:
                        raw_data = json.loads(response.text)
                        logger.info(f"Fallback to raw JSON parsing successful for {company_name}")
                        
                        # Fix key names if needed for raw JSON
                        if "Country_Region" in raw_data:
                            raw_data["Country/Region"] = raw_data.pop("Country_Region")
                        
                        return raw_data
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON from raw text: {response.text}")
                return None
            
        except Exception as e:
            logger.error(f"Error in structured data step: {str(e)}")
            return None

# Example usage
async def test_company_research():
    """Test function to demonstrate usage."""
    api_key = os.environ.get("LLM_API_KEY")
    researcher = CompanyResearcher(api_key)
    # Test with location context
    result = await researcher.research_company("Samsung Electronics", "Seoul, South Korea")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_company_research()) 