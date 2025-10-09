"""Module for extracting LinkedIn profile data."""

import time
import requests
import logging
from typing import Dict, Optional, Any

import config

logger = logging.getLogger(__name__)

def extract_linkedin_profile(
    linkedin_profile_url: str, 
    api_key: Optional[str] = None, 
    mock: bool = False
) -> Dict[str, Any]:
    """Extract LinkedIn profile data using ProxyCurl API or loads a premade JSON file.
    
    Args:
        linkedin_profile_url: The LinkedIn profile URL to extract data from.
        api_key: ProxyCurl API key. Required if mock is False.
        mock: If True, loads mock data from a premade JSON file instead of using the API.
    
    Returns:
        Dictionary containing the LinkedIn profile data.
    """
   
    try:
        if mock:
            logger.info(f"Using mock data from {config.MOCK_DATA_URL}")
            response = requests.get(config.MOCK_DATA_URL)
        if response.status_code==200:
            try:
                data = response.json()
                data = {
                    k:v
                    for k,v in data.items()
                    if v not in ([], "", None) and k not in ["people_also_viewed", "certifications"]
                }
                
                # Remove profile picture URLs from groups to clean the data
                if data.get("groups"):
                    for group_dict in data.get("groups"):
                        group_dict.pop("profile_pic_url", None)
                return data
            
            except ValueError as e:
                logger.error(f"Error parsing JSON response: {e}")
                logger.error(f"Response content: {response.text[:200]}...")  # Print first 200 chars
                return {}
            
    except Exception as e:
        logger.error(f"Error in extract_linkedin_profile: {e}")
        return {}
            
    return {}  # Replace with your implementation