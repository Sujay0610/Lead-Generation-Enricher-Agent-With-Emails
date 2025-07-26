import streamlit as st
import json
import re
import os
from typing import List, Dict, Any, Optional
import pandas as pd
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import requests
import time
from datetime import datetime, timedelta
import gspread
from google.oauth2.service_account import Credentials
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
import hashlib
from supabase import create_client
import asyncio
from simple_email_manager import SimpleEmailManager
from collections import deque

class APIKeyQueue:
    """
    A simple class to manage a rotating queue of API keys.
    It uses collections.deque to rotate keys in a round-robin fashion.
    """
    def __init__(self, keys):
        if not keys:
            raise ValueError("At least one API key must be provided.")
        self.keys = deque(keys)
        self.total_count = len(keys)

    def get_next_key(self):
        """
        Returns the next API key in a round-robin fashion and rotates the queue.
        """
        key = self.keys[0]
        self.keys.rotate(-1)
        return key
        
    def get_all_keys(self):
        """
        Returns a list of all API keys in the queue.
        """
        return list(self.keys)

    def add_key(self, key):
        """
        Adds a new API key to the queue.
        """
        self.keys.append(key)
        self.total_count += 1

    def remove_key(self, key):
        """
        Removes an API key from the queue. Raises ValueError if key is not found.
        """
        try:
            self.keys.remove(key)
            self.total_count -= 1
        except ValueError:
            raise ValueError("API key not found in the queue.")

def try_eval(x):
    """Safely evaluate a string representation of a Python object"""
    try:
        return eval(x)
    except (SyntaxError, ValueError, NameError):
        # If eval fails, return an empty list
        return []

# Configuration
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
APIFY_API_TOKENS = st.secrets.get("APIFY_API_TOKEN", [])
# Make sure APIFY_API_TOKENS is a list
if isinstance(APIFY_API_TOKENS, str):
    APIFY_API_TOKENS = [APIFY_API_TOKENS]

GOOGLE_SHEETS_CREDENTIALS = st.secrets.get("GOOGLE_SHEETS_CREDENTIALS", {})
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = st.secrets.get("GOOGLE_CSE_ID", "")
RESEND_API_KEY = st.secrets.get("RESEND_API_KEY", "")
SENDER_EMAIL = st.secrets.get("SENDER_EMAIL", "onboarding@resend.dev")  # Default Resend sender

class AIICPScorer:
    """Advanced ICP scorer using AI to analyze profile data"""
    
    def __init__(self, openai_api_key: str = None, model: str = "openai/gpt-4.1-mini"):
        # Try to get API key from parameter or environment
        self.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
            
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it via environment variable OPENAI_API_KEY or pass it as a parameter.")
            
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,
            openai_api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1/",
        )
        
        self.default_icp_prompt = """You are an ICP (Ideal Customer Profile) evaluator.

Your task is to assess how well this LinkedIn profile matches either of our two ICPs: "operations" or "field_service", using the limited structured fields available.

Profile Data:
{profile_json}

ICP Definitions:

1. Operations ICP:
- Industries: Manufacturing, Industrial Automation, Heavy Equipment, CNC, Robotics, Facility Management, Fleet Ops
- Roles (from 'jobTitle' or 'headline'): Operations Head, Plant Manager, Maintenance Lead, Production Engineer, Digital Transformation Officer
- Seniority: Manager level or above
- Company Maturity Proxy: Company founded before 2020 (≥5 years old)

2. Field Service ICP:
- Industries: Ghost kitchens, cloud kitchens, commercial real estate, managed appliances, kitchen automation, hotels
- Roles: Facility Manager, Maintenance Coordinator, Service Head, Asset Manager
- Seniority: Manager level or above
- Company Maturity Proxy: Founded before 2021 (≥3 years old)

Scoring Criteria (each 0–10):
- industry_fit: Match between 'companyIndustry' and ICP industries
- role_fit: Match between 'jobTitle' or 'headline' and ICP roles
- company_maturity_fit: Based on 'companyFoundedYear' (older = higher score)
- decision_maker: Based on 'seniority', 'functions', or leadership keywords

Scoring Weights:
- industry_fit: 30%
- role_fit: 30%
- company_maturity_fit: 20%
- decision_maker: 20%

Instructions:
- Return best-fit ICP: "operations", "field_service", or "none"
- Use strict logic; if match is weak or unclear, return "none"
- Output ONLY valid JSON (no extra explanation, markdown, or text)

Output Format:
{{
    "industry_fit": <0-10>,
    "role_fit": <0-10>,
    "company_size_fit": <0-10>,
    "decision_maker": <0-10>,
    "total_score": <weighted avg score>,
    "icp_category": "operations" | "field_service" | "none",
    "reasoning": "Brief reasoning based on the fields provided"
}}
"""


    def set_custom_prompt(self, prompt: str):
        """Set a custom ICP prompt"""
        if "{profile_json}" not in prompt:
            raise ValueError("Custom prompt must contain the {profile_json} placeholder")
        self.custom_prompt = prompt

    @property
    def icp_prompt(self):
        """Get the ICP prompt, using custom if available"""
        if hasattr(self, 'custom_prompt'):
            return self.custom_prompt
        if "icp_prompt" in st.session_state:
            return st.session_state["icp_prompt"]
        return self.default_icp_prompt

    def analyze_profile(self, profile: Dict) -> Dict:
        """Analyze a LinkedIn profile using AI to determine ICP fit"""
        try:
            # Prepare profile data for analysis - use all available fields from Apollo data
            profile_for_analysis = {
                "fullName": profile.get("fullName", ""),
                "headline": profile.get("headline", ""),
                "jobTitle": profile.get("jobTitle", ""),
                "companyName": profile.get("companyName", ""),
                "companyIndustry": profile.get("companyIndustry", ""),
                "companySize": profile.get("companySize", ""),
                "location": profile.get("location", ""),
                "city": profile.get("city", ""),
                "state": profile.get("state", ""),
                "country": profile.get("country", ""),
                "seniority": profile.get("seniority", ""),
                "departments": profile.get("departments", ""),
                "subdepartments": profile.get("subdepartments", ""),
                "functions": profile.get("functions", ""),
                "companyWebsite": profile.get("companyWebsite", ""),
                "companyDomain": profile.get("companyDomain", ""),
                "companyFoundedYear": profile.get("companyFoundedYear", ""),
                "work_experience_months": profile.get("work_experience_months", ""),
            }
            
            # Remove empty fields to reduce noise
            profile_for_analysis = {k: v for k, v in profile_for_analysis.items() if v}
            
            # Get AI analysis
            messages = [
                HumanMessage(content=self.icp_prompt.format(
                    profile_json=json.dumps(profile_for_analysis, indent=2)
                ))
            ]
            
            response = self.llm.invoke(messages)
            
            # Clean the response content to ensure it's valid JSON
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            try:
                analysis = json.loads(content)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON response from AI: {content}")
                raise Exception(f"Failed to parse AI response: {str(e)}")
            
            # Validate the analysis
            required_fields = ["industry_fit", "role_fit", "company_size_fit", "decision_maker", 
                             "total_score", "icp_category", "reasoning"]
            
            # Handle potential field mismatch between prompt and code
            if "company_maturity_fit" in analysis and "company_size_fit" not in analysis:
                analysis["company_size_fit"] = analysis["company_maturity_fit"]
            for field in required_fields:
                if field not in analysis:
                    raise Exception(f"Missing required field in AI response: {field}")
                
            # Ensure scores are numbers between 0 and 10
            score_fields = ["industry_fit", "role_fit", "company_size_fit", "decision_maker", "total_score"]
            for field in score_fields:
                score = analysis[field]
                if not isinstance(score, (int, float)) or score < 0 or score > 10:
                    raise Exception(f"Invalid score in {field}: {score}")
            
            # Calculate score percentage
            score_percentage = min(100, analysis["total_score"] * 10)
            
            # Determine grade based on score percentage
            if score_percentage >= 80:
                grade = "A+"
            elif score_percentage >= 70:
                grade = "A"
            elif score_percentage >= 60:
                grade = "B+"
            elif score_percentage >= 50:
                grade = "B"
            elif score_percentage >= 40:
                grade = "C+"
            elif score_percentage >= 30:
                grade = "C"
            else:
                grade = "D"
            
            return {
                "total_score": analysis["total_score"],
                "score_percentage": score_percentage,
                "grade": grade,
                "breakdown": {
                    "industry_fit": analysis["industry_fit"],
                    "role_fit": analysis["role_fit"],
                    "company_size_fit": analysis["company_size_fit"],
                    "decision_maker": analysis["decision_maker"],
                    "icp_category": analysis["icp_category"],
                    "reasoning": analysis["reasoning"]
                }
            }
            
        except Exception as e:
            st.error(f"Error in AI ICP scoring: {str(e)}")
            raise

class EmailGenerator:
    """AI-powered email generator for cold outreach"""
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model="openai/gpt-4.1-mini",
            temperature=0.7,  # Slightly higher temperature for more creative emails
            openai_api_key=openai_api_key,
            base_url="https://openrouter.ai/api/v1/",
        )
        
        self.default_email_prompt = """Write a short, personalized cold email for this lead:
{lead_info}

Key points:
1. Target: Operations/Maintenance leaders in manufacturing, automation, field service
2. Pain points: Manual logs, missed SLAs, reactive maintenance
3. Our solution: Real-time machine monitoring, smart alerts, automated workflows

Guidelines:
- Write the email in clean HTML format
- Use <p> for each paragraph    
- Use <br> where appropriate (e.g., in sign-offs)
- No <html> or <body> tags needed — just the inner HTML
- 2-3 short paragraphs max
- Personalize to their role/industry
- Focus on ONE relevant benefit
- Be conversational, not salesy
- End with a soft CTA

Output ONLY the HTML email body (no subject line, no markdown, no explanations)."""


    @property
    def email_prompt(self):
        """Get the email prompt, using custom if available"""
        if "email_prompt" in st.session_state:
            return st.session_state["email_prompt"]
        return self.default_email_prompt

    def generate_email(self, lead_data: Dict) -> str:
        """Generate a personalized cold email for a lead"""
        try:
            # Format lead info for the prompt
            lead_info = f"""
Name: {lead_data.get('fullName', '')}
Job Title: {lead_data.get('jobTitle', '')}
Company: {lead_data.get('companyName', '')}
Industry: {lead_data.get('companyIndustry', '')}
Location: {lead_data.get('location', '')}
About: {lead_data.get('about', '')}
LinkedIn: {lead_data.get('linkedin_url', '')}
"""
            
            messages = [
                HumanMessage(content=self.email_prompt.format(lead_info=lead_info))
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            st.error(f"Error generating email: {str(e)}")
            return None

class EmailManager:
    """Class to handle email management and sending via Resend"""
    
    def __init__(self, api_key: str, domain: str, sender_email: str):
        self.api_key = api_key
        self.sender_email = sender_email
        self.email_generator = EmailGenerator(OPENAI_API_KEY)
    
    def send_email(self, to_email: str, subject: str, body: str) -> Dict:
        """Send email using Resend API"""
        try:
            response = requests.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "from": self.sender_email,
                    "to": to_email,
                    "subject": subject,
                    "html": body  # Resend uses html parameter instead of text
                }
            )
            
            if response.status_code in [200, 201]:
                return {"status": "success", "message": "Email sent successfully"}
            else:
                return {"status": "error", "message": f"Resend API returned {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"status": "error", "message": f"Failed to send email: {str(e)}"}
    
    def generate_and_preview_email(self, lead_data: Dict) -> Dict:
        """Generate email content and return for preview"""
        try:
            # Generate email content
            email_body = self.email_generator.generate_email(lead_data)
            if not email_body:
                return {"status": "error", "message": "Failed to generate email content"}
            
            # Generate subject line
            subject = f"Quick question about {lead_data.get('companyName', 'your company')}"
            
            return {
                "status": "success",
                "subject": subject,
                "body": email_body,
                "to_email": lead_data.get("email", "")
            }
                
        except Exception as e:
            return {"status": "error", "message": f"Failed to generate email: {str(e)}"}

class LeadScrapingTool:
    """Enhanced tool for scraping leads with ICP scoring using Apollo.io or Google Search + Apify enrichment"""

    def __init__(self, apify_token_or_tokens, sheets_service, google_api_key=None, cse_id=None):
        self.sheets_service = sheets_service
        if isinstance(apify_token_or_tokens, list):
            self.token_queue = APIKeyQueue(apify_token_or_tokens)
        else:
            self.token_queue = APIKeyQueue([apify_token_or_tokens])
        self.ai_icp_scorer = AIICPScorer(OPENAI_API_KEY)
        
        # Google Search API credentials for custom search method
        self.google_api_key = google_api_key
        self.cse_id = cse_id
        
        # Initialize or load used keys from session state
        if "used_apify_keys" not in st.session_state:
            st.session_state.used_apify_keys = set()
        # Handle case where used_apify_keys might be None
        if st.session_state.used_apify_keys is None:
            st.session_state.used_apify_keys = set()
        self.used_keys = st.session_state.used_apify_keys
        
        # Track total available keys
        self.total_keys = self.token_queue.total_count
        
        # Log current key usage
        if self.used_keys:
            st.sidebar.info(f"🔑 API Key Usage: {len(self.used_keys)}/{self.total_keys} keys used today")
            
        # Initialize or load exhausted keys from session state
        if "exhausted_apify_keys" not in st.session_state:
            st.session_state.exhausted_apify_keys = set()
        # Handle case where exhausted_apify_keys might be None
        if st.session_state.exhausted_apify_keys is None:
            st.session_state.exhausted_apify_keys = set()
        self.exhausted_keys = st.session_state.exhausted_apify_keys
        
        # Initialize Supabase client
        try:
            self.supabase = create_client(
                st.secrets["SUPABASE_URL"],
                st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
            )
        except Exception as e:
            st.warning(f"Failed to initialize Supabase client: {str(e)}")
            self.supabase = None

    def get_next_unused_key(self):
        """Get the next unused API key. If all keys have been used, reset and start over."""
        # Initialize exhausted_keys if it's None
        if self.exhausted_keys is None:
            self.exhausted_keys = set()
            st.session_state.exhausted_apify_keys = self.exhausted_keys
            
        # If all keys have been used, reset the used keys tracking
        if len(self.used_keys) >= self.total_keys:
            st.warning("⚠️ All API keys have been used today. Resetting usage tracking.")
            self.used_keys = set()
            st.session_state.used_apify_keys = self.used_keys
        
        # Try to find an unused key that is not exhausted
        for _ in range(self.total_keys):
            key = self.token_queue.get_next_key()
            if key not in self.used_keys and key not in self.exhausted_keys:
                self.used_keys.add(key)
                # Update session state
                st.session_state.used_apify_keys = self.used_keys
                return key
                
        # If all keys are exhausted or used, try to find any key that's not exhausted
        if len(self.exhausted_keys) < self.total_keys:
            for _ in range(self.total_keys):
                key = self.token_queue.get_next_key()
                if key not in self.exhausted_keys:
                    return key
        
        # If we somehow get here, just return the next key
        return self.token_queue.get_next_key()
        
    def mark_key_exhausted(self, key):
        """Mark a key as exhausted for the day"""
        # Initialize exhausted_keys if it's None
        if self.exhausted_keys is None:
            self.exhausted_keys = set()
            
        if key not in self.exhausted_keys:
            self.exhausted_keys.add(key)
            st.session_state.exhausted_apify_keys = self.exhausted_keys
            st.warning(f"API key {key[:10]}... marked as exhausted for today ({len(self.exhausted_keys)}/{self.total_keys} keys exhausted)")

    def generate_apollo_url(self, query_data: dict) -> str:
        """Generate Apollo.io search URL from query parameters"""
        # Base URL for Apollo
        base_url = 'https://app.apollo.io/#/people'

        # List to hold each part of the query string
        query_parts = []

        # Add static parameters
        query_parts.append('sortByField=recommendations_score')
        query_parts.append('sortAscending=false')
        query_parts.append('page=1')

        # Helper function to process and add array parameters to query_parts
        def add_array_params(param_name: str, values: List[str]):
            for val in values:
                # Replace '+' with space then encode the value
                decoded_value = val.replace('+', ' ')
                query_parts.append(f"{param_name}[]={requests.utils.quote(decoded_value)}")

        # Process job titles (maps to personTitles[])
        if 'job_title' in query_data and isinstance(query_data['job_title'], list):
            add_array_params('personTitles', query_data['job_title'])

        # Process locations (maps to personLocations[])
        if 'location' in query_data and isinstance(query_data['location'], list):
            add_array_params('personLocations', query_data['location'])

        # Process business keywords (maps to qOrganizationKeywordTags[])
        if 'business' in query_data and isinstance(query_data['business'], list):
            add_array_params('qOrganizationKeywordTags', query_data['business'])
            
        # Process employee ranges if provided
        if 'employee_ranges' in query_data and isinstance(query_data['employee_ranges'], list):
            add_array_params('organizationNumEmployeesRanges', query_data['employee_ranges'])

        # Add static included organization keyword fields
        query_parts.append('includedOrganizationKeywordFields[]=tags')
        query_parts.append('includedOrganizationKeywordFields[]=name')

        # Only add default employee ranges if not already provided
        if 'employee_ranges' not in query_data or not isinstance(query_data['employee_ranges'], list):
            employee_ranges = [
                "1,10",
                "11,20",
                "21,50",
                "51,100",
                "101,200"
            ]
            add_array_params('organizationNumEmployeesRanges', employee_ranges)

        # Combine all query parts with '&' to form the full query string
        query_string = '&'.join(query_parts)

        # Build the final URL
        final_url = f"{base_url}?{query_string}"

        return final_url

    def search_linkedin_profiles(self, query: str, num_results: int = 20) -> List[Dict]:
        """Search for LinkedIn profile URLs via Google Custom Search."""
        results = []
        start_index = 1
        total_search_results = 0

        st.info(f"🔍 Searching Google for: {query}")

        while len(results) < num_results:
            try:
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    "key": self.google_api_key,
                    "cx": self.cse_id,
                    "q": query,
                    "start": start_index,
                    "num": min(10, num_results - len(results)),
                }

                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                # Debug: Show API response info
                if "searchInformation" in data:
                    total_results = data["searchInformation"].get("totalResults", "0")
                    st.info(f"📊 Google found {total_results} total results for this query")

                if "items" not in data:
                    st.warning(f"⚠️ No search results found in Google response")
                    break

                # Count all search results and LinkedIn profiles separately
                search_batch_size = len(data["items"])
                total_search_results += search_batch_size
                linkedin_count_in_batch = 0

                for item in data["items"]:
                    if "linkedin.com/in/" in item["link"]:
                        linkedin_count_in_batch += 1
                        results.append(
                            {
                                "title": item.get("title", ""),
                                "link": item.get("link", ""),
                                "snippet": item.get("snippet", ""),
                                "found_at": datetime.now().isoformat(),
                            }
                        )

                st.info(f"📋 Batch {start_index//10 + 1}: Found {linkedin_count_in_batch} LinkedIn profiles out of {search_batch_size} search results")

                # If no more results available, break
                if search_batch_size < 10:
                    break

                start_index += 10
                time.sleep(0.1)

            except Exception as e:
                st.error(f"❌ Error during Google search: {str(e)}")
                if "quota" in str(e).lower():
                    st.error("🚫 Google API quota exceeded. Please check your API limits.")
                elif "invalid" in str(e).lower():
                    st.error("🔑 Invalid Google API credentials. Please check your API key and CSE ID.")
                break

        if results:
            st.success(f"✅ Found {len(results)} LinkedIn profiles from {total_search_results} total search results")
        else:
            st.warning(f"⚠️ No LinkedIn profiles found despite {total_search_results} total search results. Try broader search terms.")
        
        return results

    def parse_location(self, location_str: str) -> Dict[str, str]:
        """Parse location string into city, state, and country components.
        
        Examples:
        - "Calgary, Alberta, Canada" -> {"city": "Calgary", "state": "Alberta", "country": "Canada"}
        - "Manitoba, Canada" -> {"city": "", "state": "Manitoba", "country": "Canada"}
        - "Cambridge, Ontario, Canada" -> {"city": "Cambridge", "state": "Ontario", "country": "Canada"}
        """
        result = {"city": "", "state": "", "country": ""}
        
        if not location_str:
            return result
            
        parts = [part.strip() for part in location_str.split(",")]
        
        # Handle different location formats
        if len(parts) >= 3:
            # Format: City, State/Province, Country
            result["city"] = parts[0]
            result["state"] = parts[1]
            result["country"] = parts[2]
        elif len(parts) == 2:
            # Format: State/Province, Country or City, Country
            # Assume the last part is always the country
            result["country"] = parts[1]
            
            # For the first part, we'll assume it's a state/province
            # This is a simplification - in a real system, you might want to
            # check against a database of known cities vs. states
            result["state"] = parts[0]
        elif len(parts) == 1:
            # Just a country or city name
            result["country"] = parts[0]
            
        return result
        
    def enrich_profile_with_apify(self, linkedin_url: str) -> Dict:
        """Enrich a single LinkedIn profile using Apify."""
        try:
            url = (
                "https://api.apify.com/v2/acts/dev_fusion~linkedin-profile-scraper/"
                "run-sync-get-dataset-items"
            )
            payload = {"profileUrls": [linkedin_url], "maxDelay": 5, "minDelay": 1}
            headers = {
                "Authorization": f"Bearer {self.token_queue.get_next_key()}",
                "Content-Type": "application/json",
            }

            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data:
                profile = data[0]
                
                # Parse location into city, state, and country
                location = profile.get("addressWithCountry", "")
                location_parts = self.parse_location(location)
                
                return {
                    "linkedin_url": linkedin_url,
                    "firstName": profile.get("firstName", ""),
                    "lastName": profile.get("lastName", ""),
                    "fullName": profile.get("fullName", ""),
                    "headline": profile.get("headline", ""),
                    "email": profile.get("email", ""),
                    "jobTitle": profile.get("jobTitle", ""),
                    "companyName": profile.get("companyName", ""),
                    "companyIndustry": profile.get("companyIndustry", ""),
                    "companyWebsite": profile.get("companyWebsite", ""),
                    "companyLinkedin": profile.get("companyLinkedin", ""),
                    "companySize": profile.get("companySize", ""),
                    "location": location,
                    "city": location_parts["city"],
                    "state": location_parts["state"],
                    "country": location_parts["country"],
                    "about": profile.get("about", ""),
                    "experience": json.dumps(profile.get("experiences", [])),
                    "scraped_at": datetime.now().isoformat(),
                    "scraping_status": "success"
                }
            else:
                return {
                    "linkedin_url": linkedin_url,
                    "scraping_status": "no_data",
                    "scraped_at": datetime.now().isoformat(),
                }

        except Exception as e:
            return {
                "linkedin_url": linkedin_url,
                "scraping_status": "error",
                "error_message": str(e),
                "scraped_at": datetime.now().isoformat(),
            }

    def batch_enrich_profiles(self, linkedin_urls: List[str]) -> List[Dict]:
        """Enrich multiple LinkedIn profiles with ICP scoring."""
        enriched, total = [], len(linkedin_urls)
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, url in enumerate(linkedin_urls):
            status_text.text(f"🔍 Enriching profile {i + 1}/{total}: {url}")
            profile = self.enrich_profile_with_apify(url)
            
            # Add ICP scoring if enrichment was successful
            if profile.get("scraping_status") == "success":
                icp_score = self.ai_icp_scorer.analyze_profile(profile)
                profile.update({
                    "icp_score": icp_score["total_score"],
                    "icp_percentage": icp_score["score_percentage"],
                    "icp_grade": icp_score["grade"],
                    "icp_breakdown": json.dumps(icp_score["breakdown"]),
                    "send_email_status": "Not Sent"
                })
            
            enriched.append(profile)
            progress_bar.progress((i + 1) / total)

            if i < total - 1:
                time.sleep(2)

        status_text.text(f"✅ Completed enriching {total} profiles!")
        return enriched

    def search_apollo_profiles(self, query: str, num_results: int = 50) -> List[Dict]:
        """Search for profiles using Apify Apollo scraper with API key rotation"""
        max_retries = len(self.token_queue.get_all_keys())
        
        for attempt in range(max_retries):
            try:
                current_api_key = self.token_queue.get_next_key()
                
                if attempt == 0:
                    st.info(f"🔍 Starting Apollo.io scraping with query: {query}")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text("⏳ Waiting for Apollo.io scraper to initialize (this may take a few minutes)...")
                else:
                    st.warning(f"🔄 Retrying with different API key (attempt {attempt + 1}/{max_retries})")

                # Call Apify Apollo scraper
                url = "https://api.apify.com/v2/acts/iJcISG5H8FJUSRoVA/run-sync-get-dataset-items"
                payload = {
                    "contact_email_exclude_catch_all": True,
                    "contact_email_status_v2": True,
                    "include_email": True,
                    "max_result": num_results,
                    "url": query
                }

                headers = {
                    "Authorization": f"Bearer {current_api_key}",
                    "Content-Type": "application/json"
                }

                # Make the request with a longer timeout
                response = requests.post(url, json=payload, headers=headers, timeout=600)  # 10 minute timeout
            
                # Handle both 200 and 400 responses since the scraper might return data even with 400
                if response.status_code not in [200, 201, 400]:
                    if response.status_code == 429:
                        st.warning(f"⚠️ Rate limit hit with API key {attempt + 1}. Trying next key...")
                        continue
                    elif response.status_code in [401, 403]:
                        st.warning(f"⚠️ Authentication failed with API key {attempt + 1}. Trying next key...")
                        continue
                    else:
                        st.error(f"Apollo.io API returned unexpected status code {response.status_code}")
                        if attempt == max_retries - 1:
                            return []
                        continue
                
                try:
                    results = response.json()

                    # Check if results is empty or not a list
                    if not results:
                        if attempt == max_retries - 1:
                            st.warning("No results returned from Apollo.io. The search might be too narrow or the credits might be exhausted.")
                            return []
                        continue
                    
                    # If results is a dict with an error message, check for data
                    if isinstance(results, dict):
                        if 'data' in results:
                            results = results['data']
                        elif 'items' in results:
                            results = results['items']
                        else:
                            if attempt == max_retries - 1:
                                st.warning("Unexpected response format from Apollo.io")
                                return []
                            continue
                    
                    # Ensure results is a list
                    if not isinstance(results, list):
                        if attempt == max_retries - 1:
                            st.warning("Invalid response format from Apollo.io")
                            return []
                        continue

                    st.success(f"✅ Found {len(results)} profiles from Apollo.io using API key {attempt + 1}")
                    break  # Success, exit the retry loop

                except json.JSONDecodeError:
                    if attempt == max_retries - 1:
                        st.error("Invalid JSON response from Apollo.io API")
                        return []
                    continue
                
            except requests.exceptions.Timeout:
                st.warning(f"⏱️ Request timeout with API key {attempt + 1}. Trying next key...")
                if attempt == max_retries - 1:
                    st.error("All API keys failed due to timeout")
                    return []
                continue
                
            except requests.exceptions.RequestException as e:
                st.warning(f"🔌 Request failed with API key {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    st.error(f"All API keys failed: {str(e)}")
                    return []
                continue
                
            except Exception as e:
                st.warning(f"❌ Unexpected error with API key {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    st.error(f"All API keys failed with unexpected error: {str(e)}")
                    return []
                continue
        
        # If we get here, we have successful results
        profiles = []
        for idx, result in enumerate(results):
                try:
                    # Get organization data
                    organization = result.get("organization", {}) or {}
                    
                    # Calculate work experience from employment history
                    employment_history = result.get("employment_history", [])
                    total_experience_months = 0
                    for job in employment_history:
                        try:
                            start_date = datetime.strptime(job.get("start_date", ""), "%Y-%m-%d")
                            end_date = datetime.strptime(job.get("end_date", datetime.now().strftime("%Y-%m-%d")), "%Y-%m-%d") if not job.get("current", False) else datetime.now()
                            months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                            total_experience_months += max(0, months)
                        except (ValueError, TypeError):
                            continue
                    
                    # Build location string
                    location_parts = []
                    if result.get("city"): location_parts.append(result["city"])
                    if result.get("state"): location_parts.append(result["state"])
                    if result.get("country"): location_parts.append(result["country"])
                    location = ", ".join(location_parts) if location_parts else ""
                    
                    profile = {
                        "id": result.get("id", ""),
                        "linkedin_url": result.get("linkedin_url", ""),
                        "fullName": result.get("name", ""),
                        "firstName": result.get("first_name", ""),
                        "lastName": result.get("last_name", ""),
                        "jobTitle": result.get("title", ""),
                        "email": result.get("email", ""),
                        "email_status": result.get("email_status", ""),
                        "photo_url": result.get("photo_url", ""),
                        "headline": result.get("headline", ""),
                        "location": location,
                        "city": result.get("city", ""),
                        "state": result.get("state", ""),
                        "country": result.get("country", ""),
                        "seniority": result.get("seniority", ""),
                        "departments": result.get("departments", []),
                        "subdepartments": result.get("subdepartments", []),
                        "functions": result.get("functions", []),
                        "work_experience_months": total_experience_months,
                        "employment_history": employment_history,
                        "intent_strength": result.get("intent_strength"),
                        "show_intent": result.get("show_intent", False),
                        "email_domain_catchall": result.get("email_domain_catchall", False),
                        "revealed_for_current_team": result.get("revealed_for_current_team", False),
                        
                        # Company information
                        "companyName": organization.get("name", ""),
                        "companyWebsite": organization.get("website_url", ""),
                        "companyLinkedIn": organization.get("linkedin_url", ""),
                        "companyTwitter": organization.get("twitter_url", ""),
                        "companyFacebook": organization.get("facebook_url", ""),
                        "companyPhone": organization.get("phone", ""),
                        "companyFoundedYear": organization.get("founded_year"),
                        "companySize": organization.get("size", ""),
                        "companyIndustry": organization.get("industry", ""),
                        "companyDomain": organization.get("primary_domain", ""),
                        "companyGrowth6Month": organization.get("organization_headcount_six_month_growth"),
                        "companyGrowth12Month": organization.get("organization_headcount_twelve_month_growth"),
                        "companyGrowth24Month": organization.get("organization_headcount_twenty_four_month_growth"),
                        "email_status": "Not Sent",

                    }
                    
                    profiles.append(profile)
                    
                    # Update progress
                    progress = min(1.0, (idx + 1) / len(results))
                    progress_bar.progress(progress)
                    status_text.text(f"Processing profile {idx + 1}/{len(results)}: {profile['fullName']}")
                    
                except Exception as e:
                    st.warning(f"Error processing profile {idx + 1}: {str(e)}")
                    continue

        progress_bar.empty()
        status_text.empty()
        
        if profiles:
            st.success(f"✅ Successfully processed {len(profiles)} Apollo.io profiles")
        else:
            st.warning("No profiles could be processed. Please check your search criteria.")
            
        return profiles  # Return all profiles without limiting

    def scrape_leads(self, query_json: str, method: str = "apollo", num_results: int = 20) -> str:
        """Full pipeline: search → save → score → update."""
        try:
            st.info(f"🚀 Starting lead generation process using {method.upper()} method...")

            # Step 1: Parse input and validate
            try:
                input_data = json.loads(query_json)
                if isinstance(input_data, dict) and "query" in input_data and isinstance(input_data["query"], list):
                    # Get the first non-None element from the query list
                    query_list = input_data["query"]
                    params_data = None
                    for item in query_list:
                        if item is not None:
                            params_data = item
                            break
                    
                    # If no valid item found, use empty dict
                    if params_data is None:
                        params_data = {}
                else:
                    params_data = input_data

                # Ensure params_data is not None
                if params_data is None:
                    params_data = {}

                # Add default employee ranges if not specified (for Apollo method)
                if method == "apollo" and isinstance(params_data, dict):
                    if not params_data.get("employee_ranges"):
                        params_data["employee_ranges"] = ["11,20", "21,50", "51,100", "101,200"]
                    if not params_data.get("sort_field"):
                        params_data["sort_field"] = "recommendations_score"
                    if not params_data.get("sort_ascending"):
                        params_data["sort_ascending"] = "false"

                if not isinstance(params_data, list):
                    params_data = [params_data]
                    
                # Filter out None values from the list
                params_data = [item for item in params_data if item is not None and isinstance(item, dict)]
                
                # If no valid data after filtering, return error
                if not params_data:
                    return "No valid query parameters found. Please check your input data."
                    
            except json.JSONDecodeError as e:
                return f"Invalid query JSON: {str(e)}"

            # Step 2: Scrape leads based on selected method
            all_profiles = []
            
            if method == "apollo":
                st.subheader("🔍 Step 1: Scraping Leads from Apollo.io")
                for q in params_data:
                    apollo_url = self.generate_apollo_url(q)
                    st.info(f"Generated Apollo.io search URL: {apollo_url}")
                    
                    profiles = self.search_apollo_profiles(apollo_url, num_results=num_results)
                    if profiles:
                        all_profiles.extend(profiles)
            
            elif method == "google_search":
                st.subheader("🔍 Step 1: Searching LinkedIn Profiles via Google Search")
                
                # Check if Google API credentials are available
                if not self.google_api_key or not self.cse_id:
                    return "Google Search method requires Google API Key and Custom Search Engine ID. Please configure these in your environment variables."
                
                # Generate search queries from the parameters
                linkedin_urls = []
                for q in params_data:
                    job_titles = q.get("job_title", [])
                    locations = q.get("location", [])
                    businesses = q.get("business", [])
                    

                    
                    # Since we now have single values in lists, take the first (and only) item
                    job_title = job_titles[0] if job_titles else ""
                    location = locations[0] if locations else ""
                    business = businesses[0] if businesses else ""
                    
                    # Create search terms list (only include non-empty terms)
                    search_terms = []
                    if job_title:
                        search_terms.append(job_title.replace('+', ' '))
                    if location:
                        search_terms.append(location.replace('+', ' '))
                    if business:
                        search_terms.append(business.replace('+', ' '))
                    
                    # Only proceed if we have at least one search term
                    if search_terms:
                        # Create a search query with available terms
                        search_query = f"{' '.join(search_terms)} site:linkedin.com/in/"
                        
                        st.info(f"🔍 Searching for: {search_query}")
                        
                        # Search for LinkedIn profiles
                        search_results = self.search_linkedin_profiles(search_query, num_results=num_results)
                        for result in search_results:
                            linkedin_urls.append(result["link"])
                    else:
                        st.warning("⚠️ No valid search terms found in query parameters")
                
                # Remove duplicates
                linkedin_urls = list(set(linkedin_urls))
                
                if not linkedin_urls:
                    return "No LinkedIn profiles found with the given search criteria."
                
                # Process all found LinkedIn URLs without limiting
                st.info(f"Found {len(linkedin_urls)} unique LinkedIn profiles to enrich")
                
                st.subheader("🔍 Step 2: Enriching LinkedIn Profiles with Apify")
                all_profiles = self.batch_enrich_profiles(linkedin_urls)
            
            else:
                return f"Unknown method: {method}. Supported methods are 'apollo' and 'google_search'."

            # Flatten all_profiles in case it contains nested lists
            flat_profiles = []
            for item in all_profiles:
                if isinstance(item, list):
                    flat_profiles.extend(item)
                else:
                    flat_profiles.append(item)
            all_profiles = flat_profiles

            if not all_profiles:
                return "No profiles found. Try adjusting your search criteria."

            total_leads = len(all_profiles)
            
            # Step 3: ICP Scoring
            if method == "apollo":
                st.subheader("🎯 Step 2: ICP Scoring")
                scored_profiles = []
                scoring_progress = st.progress(0)
                scoring_status = st.empty()
                
                for idx, profile in enumerate(all_profiles):
                    try:
                        scoring_status.text(f"Scoring profile {idx + 1}/{total_leads}: {profile.get('fullName', 'Unknown')}")
                        
                        # Get ICP score
                        icp_score = self.ai_icp_scorer.analyze_profile(profile)
                        
                        # Update profile with ICP score
                        profile.update({
                            "icp_score": icp_score["total_score"],
                            "icp_percentage": icp_score["score_percentage"],
                            "icp_grade": icp_score["grade"],
                            "icp_breakdown": json.dumps(icp_score["breakdown"])
                        })
                    
                        scored_profiles.append(profile)
                        scoring_progress.progress((idx + 1) / total_leads)
                        
                    except Exception as e:
                        st.warning(f"Error scoring profile {idx + 1}: {str(e)}")
                        continue
                        
                scoring_progress.empty()
                scoring_status.empty()
            else:
                # For Google Search method, ICP scoring is already done in batch_enrich_profiles
                scored_profiles = all_profiles
                st.subheader("🎯 Step 2: ICP Scoring (Completed during enrichment)")

            # Step 4: Update sheets and Supabase with scores
            st.subheader("📊 Step 3: Updating with ICP Scores")
            if scored_profiles:
                try:
                    # Save to Google Sheets
                    final_save_msg = self.save_enriched_data_to_sheets(scored_profiles)
                    st.success("✅ ICP scores saved to Google Sheets")
                    
                    # Save to Supabase
                    supabase_result = self.save_enriched_data_to_supabase(scored_profiles)
                    if supabase_result["status"] == "success":
                        st.success(f"✅ {supabase_result['message']}")
                    else:
                        st.warning(f"⚠️ {supabase_result['message']}")
                        

                        
                except Exception as e:
                    st.error(f"Error saving ICP scores: {str(e)}")
                    return "Failed to save ICP scores to Google Sheets and Supabase"

            # Final summary
            successful_scores = [p for p in scored_profiles if p.get("icp_score") is not None]
            method_name = "Apollo.io" if method == "apollo" else "Google Search + Apify"
            summary = (
                "🎉 **Lead Generation Complete!**\n\n"
                f"📊 **Results Summary:**\n"
                f"- {method_name} profiles found: {total_leads}\n"
                f"- Successfully scored: {len(successful_scores)}\n"
            )
            
            if successful_scores:
                avg_score = sum(float(p["icp_percentage"]) for p in successful_scores) / len(successful_scores)
                grade_counts = {}
                for p in successful_scores:
                    grade = p.get("icp_grade", "Unknown")
                    grade_counts[grade] = grade_counts.get(grade, 0) + 1
                
                summary += f"- Average ICP Score: {avg_score:.1f}%\n"
                summary += f"- Grade Distribution: {grade_counts}\n"
            
            return summary

        except Exception as e:
            import traceback, sys
            traceback.print_exc(file=sys.stderr)
            return f"Error in lead generation process: {str(e)}"
            
    def map_profile_fields_to_db(self, profile: Dict) -> Dict:
        """Map camelCase profile fields to snake_case database column names."""
        # Field mapping from camelCase (Apollo/Apify) to snake_case (database)
        field_mapping = {
            'fullName': 'full_name',
            'firstName': 'first_name', 
            'lastName': 'last_name',
            'jobTitle': 'job_title',
            'companyName': 'company_name',
            'companyDomain': 'company_domain',
            'companyIndustry': 'company_industry',
            'companySize': 'company_size',
            'companyWebsite': 'company_website',
            'companyLinkedIn': 'company_linkedin',
            'companyLinkedin': 'company_linkedin',  # Handle both variations
            'companyTwitter': 'company_twitter',
            'companyFacebook': 'company_facebook',
            'companyPhone': 'company_phone',
            'companyFoundedYear': 'company_founded_year',
            'companyFoundedIn': 'company_founded_year',  # Handle LinkedIn variation
            'companyGrowth6Month': 'company_growth_6month',
            'companyGrowth12Month': 'company_growth_12month',
            'companyGrowth24Month': 'company_growth_24month',
            'linkedinUrl': 'linkedin_url',
            'linkedin_url': 'linkedin_url',
            'emailAddress': 'email_address',
            'phoneNumber': 'phone_number',
            'mobileNumber': 'phone_number',  # Handle LinkedIn variation
            'photo_url': 'photo_url',
            'profilePic': 'photo_url',  # Handle LinkedIn variation
            'profilePicHighQuality': 'photo_url',  # Use high quality if available
            'work_experience_months': 'work_experience_months',
            'employment_history': 'employment_history',
            'intent_strength': 'intent_strength',
            'show_intent': 'show_intent',
            'email_domain_catchall': 'email_domain_catchall',
            'revealed_for_current_team': 'revealed_for_current_team',
            'icpScore': 'icp_score',
            'icpGrade': 'icp_grade', 
            'icpPercentage': 'icp_percentage',
            'icpBreakdown': 'icp_breakdown',
            'icp_score': 'icp_score',
            'icp_grade': 'icp_grade',
            'icp_percentage': 'icp_percentage',
            'icp_breakdown': 'icp_breakdown',
            'createdAt': 'created_at',
            # LinkedIn enrichment specific fields
            'about': 'about',
            'headline': 'headline',
            'experiences': 'experience',  # Map experiences array to experience JSON field
            'connections': 'connections',  # This might not be in DB schema, will be filtered out
            'followers': 'followers',  # This might not be in DB schema, will be filtered out
            'currentJobDuration': 'current_job_duration',  # This might not be in DB schema
            'currentJobDurationInYrs': 'current_job_duration_years',  # This might not be in DB schema
            'topSkillsByEndorsements': 'top_skills',  # This might not be in DB schema
            'addressCountryOnly': 'country',
            'addressWithCountry': 'location',
            'addressWithoutCountry': 'location',
            'publicIdentifier': 'public_identifier',  # This might not be in DB schema
            'openConnection': 'open_connection',  # This might not be in DB schema
            'urn': 'linkedin_urn'  # This might not be in DB schema
        }
        
        mapped_profile = {}
        for key, value in profile.items():
            # Use mapped field name if available, otherwise use original key in snake_case
            db_field = field_mapping.get(key, key)
            
            # Special handling for specific fields
            if key == 'profilePicHighQuality' and value and 'profilePic' in profile:
                # Prefer high quality profile pic over regular one
                db_field = 'photo_url'
            elif key == 'profilePic' and 'profilePicHighQuality' in profile and profile['profilePicHighQuality']:
                # Skip regular profile pic if high quality is available
                continue
            
            # Convert lists and dicts to JSON strings
            if isinstance(value, (list, dict)):
                mapped_profile[db_field] = json.dumps(value)
            # Convert None to empty string for text fields
            elif value is None:
                mapped_profile[db_field] = ""
            # Handle string values that might contain backticks (clean them)
            elif isinstance(value, str):
                # Remove backticks and extra spaces from URLs and other string fields
                cleaned_value = value.strip().strip('`').strip()
                mapped_profile[db_field] = cleaned_value
            # Keep other values as is
            else:
                mapped_profile[db_field] = value
                
        return mapped_profile
    
    def get_valid_db_columns(self):
        """Get list of valid columns from the leads table schema."""
        # Define the current leads table schema columns based on the complete migration
        # This matches the 20241201_create_complete_leads_table.sql schema
        base_columns = {
            'id', 'linkedin_url', 'full_name', 'first_name', 'last_name', 'headline', 'about',
            'email', 'email_address', 'email_status', 'phone_number', 'job_title', 'seniority',
            'departments', 'subdepartments', 'functions', 'work_experience_months', 'employment_history',
            'location', 'city', 'state', 'country', 'company_name', 'company_website', 'company_domain',
            'company_linkedin', 'company_twitter', 'company_facebook', 'company_phone', 'company_size',
            'company_industry', 'company_founded_year', 'company_growth_6month', 'company_growth_12month',
            'company_growth_24month', 'photo_url', 'experience', 'intent_strength', 'show_intent',
            'email_domain_catchall', 'revealed_for_current_team', 'icp_score', 'icp_percentage',
            'icp_grade', 'icp_breakdown', 'send_email_status', 'scraping_status', 'error_message',
            'scraped_at', 'created_at', 'updated_at'
        }
        
        # Additional columns that might be added in future or custom implementations
        # These will be filtered out gracefully if they don't exist
        extended_columns = {
            'connections', 'followers', 'current_job_duration', 'current_job_duration_years',
            'top_skills', 'public_identifier', 'open_connection', 'linkedin_urn'
        }
        
        return base_columns.union(extended_columns)

    def save_enriched_data_to_supabase(self, enriched_data: List[Dict]) -> Dict:
        """Save enriched data to Supabase leads table with graceful handling of missing columns."""
        if not enriched_data:
            return {"status": "warning", "message": "No enriched data to save to Supabase."}
            
        if not self.supabase:
            return {"status": "warning", "message": "Supabase client not initialized. Data saved only to Google Sheets."}
            
        try:
            # Get valid database columns
            valid_columns = self.get_valid_db_columns()
            
            # Track stats for summary
            total_leads = len(enriched_data)
            successful_inserts = 0
            duplicate_leads = 0
            failed_inserts = 0
            skipped_fields = set()
            
            for profile in enriched_data:
                try:
                    # Map profile fields to database column names
                    clean_profile = self.map_profile_fields_to_db(profile)
                    
                    # Filter out fields that don't exist in the database schema
                    filtered_profile = {}
                    for key, value in clean_profile.items():
                        if key in valid_columns:
                            filtered_profile[key] = value
                        else:
                            skipped_fields.add(key)
                    
                    # Add created_at timestamp if not present
                    if "created_at" not in filtered_profile:
                        filtered_profile["created_at"] = datetime.now().isoformat()
                        
                    # Insert data into Supabase
                    result = self.supabase.table("leads").insert(filtered_profile).execute()
                    
                    # Check if insert was successful
                    if result.data:
                        successful_inserts += 1
                    else:
                        failed_inserts += 1
                        
                except Exception as profile_e:
                    # Check if it's a duplicate key violation
                    if "duplicate key" in str(profile_e).lower() or "unique constraint" in str(profile_e).lower():
                        duplicate_leads += 1
                    else:
                        failed_inserts += 1
                        st.warning(f"Error inserting profile: {str(profile_e)}")
            
            # Generate summary message
            message = f"Saved {successful_inserts} new leads to Supabase"
            if duplicate_leads > 0:
                message += f", {duplicate_leads} duplicates skipped"
            if failed_inserts > 0:
                message += f", {failed_inserts} failed"
            if skipped_fields:
                st.info(f"Note: Skipped fields not in database schema: {', '.join(sorted(skipped_fields))}")
                
            return {
                "status": "success" if successful_inserts > 0 else "warning",
                "message": message,
                "stats": {
                    "total": total_leads,
                    "successful": successful_inserts,
                    "duplicates": duplicate_leads,
                    "failed": failed_inserts,
                    "skipped_fields": list(skipped_fields)
                }
            }
                
        except Exception as e:
            return {"status": "warning", "message": f"Error saving to Supabase: {str(e)}"}
    


    def save_enriched_data_to_sheets(self, enriched_data: List[Dict], sheet_name: str = "Sheet1") -> str:
        """Append enriched data to Google Sheets with ICP scoring."""
        if not enriched_data:
            return "No enriched data to save."

        try:
            ss = self.sheets_service.open("Leadgen")
            sheet = ss.worksheet(sheet_name)

            # Define expected columns in the correct order
            expected_columns = [
                "linkedin_url", "fullName", "firstName", "lastName", "email", "email_status",
                "jobTitle", "headline", "location", "city", "state", "country",
                "companyName", "companyWebsite", "companyLinkedIn", "companyTwitter",
                "companyFacebook", "companyPhone", "companySize", "companyIndustry",
                "companyDomain", "companyFoundedYear", "companyGrowth6Month",
                "companyGrowth12Month", "companyGrowth24Month", "seniority",
                "departments", "subdepartments", "functions", "work_experience_months",
                "employment_history", "intent_strength", "show_intent",
                "email_domain_catchall", "revealed_for_current_team", "photo_url",
                 "icp_score", "icp_percentage",
                "icp_grade", "icp_breakdown",
                "email_status"
            ]

            # Convert enriched data to DataFrame
            df_new = pd.DataFrame(enriched_data)
            
            # Ensure all expected columns exist
            for col in expected_columns:
                if col not in df_new.columns:
                    df_new[col] = ""  # Add missing columns with empty values

            # Reorder columns to match expected order
            df_new = df_new[expected_columns]
            
            # Clean the data
            for col in df_new.columns:
                # Convert lists and dicts to strings
                if df_new[col].apply(lambda x: isinstance(x, (list, dict))).any():
                    df_new[col] = df_new[col].apply(lambda x: str(x) if x else "")
                
                # Convert None to empty string
                df_new[col] = df_new[col].fillna("")
                
                # Convert all values to strings
                df_new[col] = df_new[col].astype(str)

            # Get existing data from sheet
            current_values = sheet.get_all_values()
            
            if not current_values:
                # If sheet is empty, add headers
                sheet.append_row(expected_columns)
                existing_keys = set()
            else:
                # Verify headers match - be more flexible with header comparison
                existing_headers = current_values[0]
                # Check if essential columns exist rather than exact match
                essential_columns = ['linkedin_url', 'fullName', 'email', 'jobTitle', 'companyName']
                headers_compatible = all(col in existing_headers for col in essential_columns)
                
                if not headers_compatible:
                    # Only clear if essential columns are missing
                    st.warning("Sheet headers don't contain essential columns. Updating headers...")
                    sheet.clear()
                    sheet.append_row(expected_columns)
                    existing_keys = set()
                else:
                    # Get existing linkedin_urls for deduplication
                    if len(current_values) > 1:
                        try:
                            existing_rows = pd.DataFrame(current_values[1:], columns=existing_headers)
                            linkedin_col = 'linkedin_url' if 'linkedin_url' in existing_headers else None
                            if linkedin_col:
                                existing_keys = set(existing_rows[linkedin_col].str.strip().str.lower())
                            else:
                                existing_keys = set()
                        except Exception as e:
                            st.warning(f"Error processing existing data: {str(e)}")
                            existing_keys = set()
                    else:
                        existing_keys = set()

            # Clean up new data and filter out duplicates
            df_new["linkedin_url"] = df_new["linkedin_url"].str.strip().str.lower()
            df_to_upload = df_new[~df_new["linkedin_url"].isin(existing_keys)]

            if df_to_upload.empty:
                return "All profiles already exist in the sheet. Nothing new to append."

            # Convert DataFrame to list of lists for upload
            values_to_upload = df_to_upload.values.tolist()

            # Append new data in batches to avoid API limits
            batch_size = 50
            for i in range(0, len(values_to_upload), batch_size):
                batch = values_to_upload[i:i + batch_size]
                sheet.append_rows(
                    batch,
                    value_input_option="RAW",
                    insert_data_option="INSERT_ROWS"
                )

            # Generate summary
            if not df_to_upload.empty and 'icp_percentage' in df_to_upload.columns:
                avg_score = pd.to_numeric(df_to_upload['icp_percentage'], errors='coerce').mean()
                grade_counts = df_to_upload['icp_grade'].value_counts().to_dict() if 'icp_grade' in df_to_upload.columns else {}
                
                summary = f"Successfully appended {len(df_to_upload)} new enriched profiles!\n"
                if not pd.isna(avg_score):
                    summary += f"Average ICP Score: {avg_score:.1f}%\n"
                if grade_counts:
                    summary += f"Grade Distribution: {grade_counts}"
                return summary

            return f"Successfully appended {len(df_to_upload)} new enriched profiles to Google Sheets!"

        except Exception as e:
            import traceback, sys
            traceback.print_exc(file=sys.stderr)
            return f"Error appending enriched data to sheets: {str(e)}"

def setup_google_sheets():
    """Setup Google Sheets service"""
    try:
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        
        creds = Credentials.from_service_account_info(GOOGLE_SHEETS_CREDENTIALS, scopes=scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"Error setting up Google Sheets: {str(e)}")
        return None

def get_leads_from_sheets(sheets_service) -> pd.DataFrame:
    """Get all leads from Google Sheets"""
    try:
        # Try to open the sheet, with error handling
        try:
            ss = sheets_service.open("Leadgen")
        except Exception as e:
            st.error(f"Could not open 'Leadgen' spreadsheet: {str(e)}")
            # Try to list available spreadsheets
            try:
                available_sheets = [sheet.title for sheet in sheets_service.openall()]
                if available_sheets:
                    st.info(f"Available spreadsheets: {', '.join(available_sheets)}")
                else:
                    st.warning("No spreadsheets found in this Google account")
            except Exception as list_e:
                st.error(f"Could not list available spreadsheets: {str(list_e)}")
            return pd.DataFrame()
            
        try:
            sheet = ss.worksheet("Sheet1")
        except Exception as e:
            st.error(f"Could not open 'Sheet1' worksheet: {str(e)}")
            # Try to list available worksheets
            try:
                available_worksheets = [ws.title for ws in ss.worksheets()]
                if available_worksheets:
                    st.info(f"Available worksheets: {', '.join(available_worksheets)}")
                else:
                    st.warning("No worksheets found in the 'Leadgen' spreadsheet")
            except Exception as list_e:
                st.error(f"Could not list available worksheets: {str(list_e)}")
            return pd.DataFrame()
        
        # Try alternative approach using get_all_records
        try:
            # This method automatically handles headers and returns a list of dictionaries
            records = sheet.get_all_records()
            if not records:
                st.warning("No records found in Google Sheet")
                return pd.DataFrame()
                
            # Create DataFrame directly from records
            df = pd.DataFrame(records)
            
        except Exception as e:
            st.warning(f"Error getting records from Google Sheet: {str(e)}")
            
            # Fall back to original approach
            try:
                all_values = sheet.get_all_values()
                
                # Check if all_values is a Response object
                if hasattr(all_values, 'status_code'):
                    st.warning(f"Received Response object instead of values: {all_values}")
                    return pd.DataFrame()
                    
                if not all_values or len(all_values) < 2:  # No data or only headers
                    return pd.DataFrame()
                    
                # Get headers and data
                headers = all_values[0]
                data = all_values[1:]
                
                # Create DataFrame
                df = pd.DataFrame(data, columns=headers)
            except Exception as nested_e:
                st.error(f"Failed to get sheet values: {str(nested_e)}")
                return pd.DataFrame()
        
        # Convert column names to lowercase for case-insensitive matching
        df.columns = df.columns.str.lower()
        
        # Debug info (only show if there are rows)
        if len(df) > 0:
            st.success(f"Successfully loaded {len(df)} rows from Google Sheet")
        
                    # Skip filtering by scraping_status since the column might not exist
            if not df.empty and 'scraping_status' in df.columns:
                try:
                    # Convert to string first to handle non-string values
                    df['scraping_status'] = df['scraping_status'].astype(str)
                    df = df[df['scraping_status'].str.lower() == 'success']
                except Exception as e:
                    st.warning(f"Error filtering by scraping_status: {str(e)}")
                    # If filtering fails, keep all rows
            # Otherwise, keep all leads
            
            # Convert numeric columns to proper types
            numeric_columns = ['icp_percentage', 'icp_score', 'companygrowth6month', 
                             'companygrowth12month', 'companygrowth24month', 'work_experience_months']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert list/dict string representations back to objects
            list_columns = ['departments', 'subdepartments', 'functions', 'employment_history']
            for col in list_columns:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: 
                        [] if not x or not x.strip() or x.startswith("<Response") 
                        else (
                            try_eval(x)
                        )
                    )
            
            # Rename columns to match expected names (if needed)
            column_mapping = {
                'linkedin_url': 'linkedin_url',
                'fullname': 'fullName',
                'firstname': 'firstName',
                'lastname': 'lastName',
                'jobtitle': 'jobTitle',
                'companyname': 'companyName',
                'companywebsite': 'companyWebsite',
                'companylinkedin': 'companyLinkedIn',
                'companytwitter': 'companyTwitter',
                'companyfacebook': 'companyFacebook',
                'companyphone': 'companyPhone',
                'companysize': 'companySize',
                'companyindustry': 'companyIndustry',
                'companydomain': 'companyDomain',
                'companyfoundedyear': 'companyFoundedYear',
                'email_status': 'email_status',
                'photo_url': 'photo_url',
                'headline': 'headline',
                'location': 'location',
                'city': 'city',
                'state': 'state',
                'country': 'country',
                'seniority': 'seniority',
                'intent_strength': 'intent_strength',
                'show_intent': 'show_intent',
                'email_domain_catchall': 'email_domain_catchall',
                'revealed_for_current_team': 'revealed_for_current_team',
                'icp_score': 'icp_score',
                'icp_percentage': 'icp_percentage',
                'icp_grade': 'icp_grade',
                'icp_breakdown': 'icp_breakdown',
                'email_status': 'email_status'
            }
            
            # Only rename columns that exist
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Convert boolean columns
            bool_columns = ['show_intent', 'email_domain_catchall', 'revealed_for_current_team']
            for col in bool_columns:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: 
                        True if str(x).lower() == 'true' 
                        else False if str(x).lower() == 'false' 
                        else None)
        
        return df
        
    except Exception as e:
        st.error(f"Error reading leads from sheets: {str(e)}")
        return pd.DataFrame()

def email_management_page():
    """Email management page"""
    st.header("📧 Email Management")
    st.markdown("Review your leads and send personalized cold emails")
    
    # Get leads from sheets
    sheets_service = setup_google_sheets()
    if not sheets_service:
        st.error("Unable to connect to Google Sheets")
        return
    
    leads_df = get_leads_from_sheets(sheets_service)
    
    if leads_df.empty:
        st.info("No leads found. Please run the lead generation process first.")
        return
    
    # Filters
    st.subheader("🎯 Filter & Sort Leads")
    
    # First row of filters
    col1, col2 = st.columns(2)
    
    with col1:
        min_score = st.slider("Minimum ICP Score (%)", 0, 100, 0)
    
    with col2:
        email_status_filter = st.selectbox(
            "Email Status",
            options=["All", "Sent", "Not Sent"],
            index=0
        )
    
    # Second row of filters - Geographic filters
    col3, col4, col5 = st.columns(3)
    
    with col3:
        # Get unique cities for filter options
        unique_cities = ["All"] + sorted(leads_df['city'].dropna().unique().tolist()) if 'city' in leads_df.columns else ["All"]
        city_filter = st.selectbox(
            "🏙️ City",
            options=unique_cities,
            index=0
        )
    
    with col4:
        # Get unique states for filter options
        unique_states = ["All"] + sorted(leads_df['state'].dropna().unique().tolist()) if 'state' in leads_df.columns else ["All"]
        state_filter = st.selectbox(
            "🗺️ State",
            options=unique_states,
            index=0
        )
    
    with col5:
        # Get unique countries for filter options
        unique_countries = ["All"] + sorted(leads_df['country'].dropna().unique().tolist()) if 'country' in leads_df.columns else ["All"]
        country_filter = st.selectbox(
            "🌍 Country",
            options=unique_countries,
            index=0
        )
    
    # Third row of filters - Job Title
    col6, col7 = st.columns([1, 1])
    
    with col6:
        # Get unique job titles for filter options
        unique_job_titles = ["All"] + sorted(leads_df['jobTitle'].dropna().unique().tolist()) if 'jobTitle' in leads_df.columns else ["All"]
        job_title_filter = st.selectbox(
            "👔 Job Title",
            options=unique_job_titles,
            index=0
        )
    
    # Sorting options
    sort_by = st.selectbox(
        "Sort by",
        options=["ICP Score (Best to Worst)", "ICP Score (Worst to Best)"],
        index=0
    )

    # # Bulk email section
    # st.markdown("---")
    # st.subheader("📨 Bulk Email Options")
    
    # # Initialize email manager
    # email_manager = EmailManager(RESEND_API_KEY, "resend.dev", SENDER_EMAIL)
    
    # # Calculate stats for unsent emails
    # unsent_leads = []
    # for idx, lead in leads_df.iterrows():
    #     email_key = f"email_sent_{hashlib.md5(lead.get('linkedin_url', '').encode()).hexdigest()}"
    #     if not st.session_state.get(email_key, False):
    #         unsent_leads.append(lead)
    
    # col1, col2 = st.columns([3, 1])
    # with col1:
    #     st.info(f"📊 {len(unsent_leads)} leads haven't been emailed yet")
        
    # with col2:
    #     if len(unsent_leads) > 0:
    #         if st.button("🚀 Generate & Send All", help="Generate and send emails to all leads that haven't been contacted"):
    #             bulk_progress = st.progress(0)
    #             status_text = st.empty()
                
    #             successful_sends = 0
    #             failed_sends = 0
                
    #             for idx, lead in enumerate(unsent_leads):
    #                 try:
    #                     status_text.text(f"Processing {idx + 1}/{len(unsent_leads)}: {lead.get('fullName', 'Unknown')}")
                        
    #                     # Prepare lead data
    #                     lead_data = {
    #                         "fullName": lead.get('fullName', ''),
    #                         "firstName": lead.get('firstName', ''),
    #                         "lastName": lead.get('lastName', ''),
    #                         "email": lead.get('email', ''),
    #                         "jobTitle": lead.get('jobTitle', ''),
    #                         "companyName": lead.get('companyName', ''),
    #                         "companyIndustry": lead.get('companyIndustry', ''),
    #                         "companyWebsite": lead.get('companyWebsite', ''),
    #                         "companyLinkedin": lead.get('companyLinkedin', ''),
    #                         "linkedin_url": lead.get('linkedin_url', ''),
    #                         "location": lead.get('location', ''),
    #                         "headline": lead.get('headline', ''),
    #                         "about": lead.get('about', ''),
    #                         "icp_score": lead.get('icp_percentage', 0),
    #                         "icp_grade": lead.get('icp_grade', 'N/A')
    #                     }
                        
    #                     # Generate email
    #                     result = email_manager.generate_and_preview_email(lead_data)
                        
    #                     if result["status"] == "success":
    #                         # Send email
    #                         send_result = email_manager.send_email(
    #                             result["to_email"],
    #                             result["subject"],
    #                             result["body"]
    #                         )
                            
    #                         if send_result["status"] == "success":
    #                             # Mark as sent in session state
    #                             email_key = f"email_sent_{hashlib.md5(lead.get('linkedin_url', '').encode()).hexdigest()}"
    #                             st.session_state[email_key] = True
                                
    #                             # Update the email_status in Google Sheets
    #                             try:
    #                                 # Get the sheet
    #                                 ss = sheets_service.open("Leadgen")
    #                                 sheet = ss.worksheet("Sheet1")
                                    
    #                                 # Find the row with this LinkedIn URL
    #                                 linkedin_url = lead.get('linkedin_url', '')
    #                                 if linkedin_url:
    #                                     # Get all LinkedIn URLs
    #                                     linkedin_urls = sheet.col_values(1)  # Assuming LinkedIn URL is in column A
    #                                     if linkedin_url in linkedin_urls:
    #                                         row_idx = linkedin_urls.index(linkedin_url) + 1  # +1 because sheets are 1-indexed
    #                                         # Find the email_status column
    #                                         headers = sheet.row_values(1)
    #                                         if 'email_status' in headers:
    #                                             col_idx = headers.index('email_status') + 1  # +1 because sheets are 1-indexed
    #                                             # Update the cell
    #                                             sheet.update_cell(row_idx, col_idx, "Sent")
    #                             except Exception as sheet_e:
    #                                 st.warning(f"Could not update email status in sheet: {str(sheet_e)}")
                                
    #                             successful_sends += 1
    #                         else:
    #                             failed_sends += 1
    #                     else:
    #                         failed_sends += 1
                        
    #                     # Update progress
    #                     bulk_progress.progress((idx + 1) / len(unsent_leads))
                        
    #                 except Exception as e:
    #                     st.error(f"Error processing {lead.get('fullName', 'Unknown')}: {str(e)}")
    #                     failed_sends += 1
                    
    #                 # Small delay to avoid rate limits
    #                 time.sleep(1)
                
    #             # Show final results
    #             if successful_sends > 0:
    #                 st.success(f"✅ Successfully sent {successful_sends} emails!")
    #             if failed_sends > 0:
    #                 st.error(f"❌ Failed to send {failed_sends} emails")
                    
    #             # Clear progress
    #             bulk_progress.empty()
    #             status_text.empty()
                
    #             # Refresh the page to update UI
    #             st.rerun()
    
    # Display leads in a compact format
    st.markdown("---")
    st.subheader(f"📋 Individual Lead Management")
    
    email_manager = EmailManager(RESEND_API_KEY, "resend.dev", SENDER_EMAIL)
    
    # Filter out leads with missing or empty emails
    leads_df = leads_df[leads_df['email'].notna() & (leads_df['email'].str.strip() != '')]

    # Apply filters
    filtered_df = leads_df.copy()
    
    # Filter by ICP score
    if 'icp_percentage' in filtered_df.columns:
        try:
            # Convert to numeric first, coercing errors to NaN
            filtered_df['icp_percentage'] = pd.to_numeric(filtered_df['icp_percentage'], errors='coerce')
            # Filter out NaN values and apply min_score filter
            filtered_df = filtered_df[filtered_df['icp_percentage'].notna() & (filtered_df['icp_percentage'] >= min_score)]
        except Exception as e:
            st.warning(f"Error filtering by ICP score: {str(e)}")
    
    # Filter by email status
    if email_status_filter != "All" and 'email_status' in filtered_df.columns:
        try:
            # Convert to string first to handle non-string values
            filtered_df['email_status'] = filtered_df['email_status'].astype(str)
            
            if email_status_filter == "Sent":
                filtered_df = filtered_df[filtered_df['email_status'].str.lower() == 'sent']
            elif email_status_filter == "Not Sent":
                # Include both "Not Sent" and empty values
                filtered_df = filtered_df[
                    (filtered_df['email_status'].str.lower() == 'not sent') | 
                    (filtered_df['email_status'].str.strip() == '')
                ]
        except Exception as e:
            st.warning(f"Error filtering by email status: {str(e)}")
    
    # Filter by city
    if city_filter != "All" and 'city' in filtered_df.columns:
        try:
            # Convert to string first to handle non-string values
            filtered_df['city'] = filtered_df['city'].astype(str)
            filtered_df = filtered_df[filtered_df['city'] == city_filter]
        except Exception as e:
            st.warning(f"Error filtering by city: {str(e)}")
    
    # Filter by state
    if state_filter != "All" and 'state' in filtered_df.columns:
        try:
            # Convert to string first to handle non-string values
            filtered_df['state'] = filtered_df['state'].astype(str)
            filtered_df = filtered_df[filtered_df['state'] == state_filter]
        except Exception as e:
            st.warning(f"Error filtering by state: {str(e)}")
    
    # Filter by country
    if country_filter != "All" and 'country' in filtered_df.columns:
        try:
            # Convert to string first to handle non-string values
            filtered_df['country'] = filtered_df['country'].astype(str)
            filtered_df = filtered_df[filtered_df['country'] == country_filter]
        except Exception as e:
            st.warning(f"Error filtering by country: {str(e)}")
    
    # Filter by job title
    if job_title_filter != "All" and 'jobTitle' in filtered_df.columns:
        try:
            # Convert to string first to handle non-string values
            filtered_df['jobTitle'] = filtered_df['jobTitle'].astype(str)
            filtered_df = filtered_df[filtered_df['jobTitle'] == job_title_filter]
        except Exception as e:
            st.warning(f"Error filtering by job title: {str(e)}")
    
    # Apply sorting
    try:
        if sort_by == "ICP Score (Best to Worst)" and 'icp_percentage' in filtered_df.columns:
            # Ensure icp_percentage is numeric
            filtered_df['icp_percentage'] = pd.to_numeric(filtered_df['icp_percentage'], errors='coerce')
            filtered_df = filtered_df.sort_values('icp_percentage', ascending=False, na_position='last')
        elif sort_by == "ICP Score (Worst to Best)" and 'icp_percentage' in filtered_df.columns:
            filtered_df['icp_percentage'] = pd.to_numeric(filtered_df['icp_percentage'], errors='coerce')
            filtered_df = filtered_df.sort_values('icp_percentage', ascending=True, na_position='last')
    except Exception as e:
        st.warning(f"Error sorting leads: {str(e)}")
    
    # Display filter summary
    total_leads = len(leads_df)
    filtered_leads = len(filtered_df)
    
    if filtered_leads < total_leads:
        st.info(f"📊 Showing {filtered_leads} of {total_leads} leads (filtered)")
    else:
        st.info(f"📊 Showing all {total_leads} leads")
    
    if filtered_df.empty:
        st.warning("No leads match your current filters. Try adjusting your filter criteria.")
        return
    
    # Create a container for the leads
    leads_container = st.container()
    
    with leads_container:
        for idx, lead in filtered_df.iterrows():
            # Create a horizontal line between leads
            if idx > 0:
                st.markdown("---")
            
            # Main lead info row
            col1, col2, col3 = st.columns([2.5, 1.5, 2])
            
            with col1:
                name = lead.get('fullName', 'Unknown')
                title = lead.get('jobTitle', 'Unknown Title')
                company = lead.get('companyName', 'Unknown Company')
                industry = lead.get('companyIndustry', 'Unknown Industry')
                linkedin_url = lead.get('linkedin_url', '')
                
                # Display name with LinkedIn link if available
                if linkedin_url:
                    st.markdown(f"**[{name}]({linkedin_url})** • {title}")
                else:
                    st.markdown(f"**{name}** • {title}")
                    
                st.markdown(f"🏢 **{company}**")
                
                # Build location string from city, state, country
                location_parts = []
                if lead.get('city') and str(lead.get('city')).strip():
                    location_parts.append(str(lead.get('city')).strip())
                if lead.get('state') and str(lead.get('state')).strip():
                    location_parts.append(str(lead.get('state')).strip())
                if lead.get('country') and str(lead.get('country')).strip():
                    location_parts.append(str(lead.get('country')).strip())
                
                location_display = ', '.join(location_parts) if location_parts else 'N/A'
                st.markdown(f"🏭 {industry} • 📍 {location_display}")
                
                # Add company contact information
                with st.expander("📞 Company Contact Info"):
                    contact_info = {
                        "🌐 Website": lead.get('companyWebsite', 'N/A'),
                        "💼 LinkedIn": lead.get('companyLinkedIn', 'N/A'),
                        "🐦 Twitter": lead.get('companyTwitter', 'N/A'),
                        "👥 Facebook": lead.get('companyFacebook', 'N/A'),
                        "📞 Phone": lead.get('companyPhone', 'N/A')
                    }
                    
                    for label, value in contact_info.items():
                        if value != 'N/A' and value.strip():
                            if any(domain in value.lower() for domain in ['http', 'www', 'linkedin', 'twitter', 'facebook']):
                                st.markdown(f"{label}: [{value}]({value})")
                            else:
                                st.markdown(f"{label}: {value}")
            
            with col2:
                icp_score = lead.get('icp_percentage', 0)
                icp_grade = lead.get('icp_grade', 'N/A')
                st.markdown(f"**ICP:** {icp_score}% ({icp_grade})")
                st.markdown(f"📧 {lead.get('email', 'No email')}")
            
            with col3:
                # Email status tracking
                email_key = f"email_sent_{hashlib.md5(lead.get('linkedin_url', '').encode()).hexdigest()}"
                compose_key = f"compose_{email_key}"
                
                if st.session_state.get(email_key, False):
                    st.success("✅ Sent")
                    if st.button("📧 Send Again", key=f"resend_{idx}", help="Send another email"):
                        st.session_state[email_key] = False
                        st.session_state[compose_key] = False
                        st.rerun()
                else:
                    if not st.session_state.get(compose_key, False):
                        if st.button("📝 Compose Email", key=f"compose_{idx}"):
                            st.session_state[compose_key] = True
                            st.rerun()
                    else:
                        # Show email composition form
                        st.markdown("#### 📨 Compose Email")
                        
                        # Add template selection
                        template_col1, template_col2 = st.columns([2, 2])
                        with template_col1:
                            persona = st.selectbox(
                                "Select Persona",
                                ["operations_manager", "facility_manager", "maintenance_manager", "plant_manager"],
                                key=f"persona_{idx}"
                            )
                        with template_col2:
                            stage = st.selectbox(
                                "Email Stage",
                                ["initial_outreach", "follow_up", "meeting_request"],
                                key=f"stage_{idx}"
                            )
                        
                        # Add buttons for template and draft actions
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            if st.button("🔄 Use Template", key=f"template_{idx}"):
                                with st.spinner("Generating from template..."):
                                    try:
                                        # Get templates and generate email
                                        template_manager = SimpleEmailManager()
                                        templates = asyncio.run(template_manager.retrieve_templates(persona, stage))
                                        if templates:
                                            result = asyncio.run(template_manager.generate_email(lead, templates))
                                            if result["status"] == "success":
                                                st.session_state[f"subject_{idx}"] = result["subject"]
                                                st.session_state[f"body_{idx}"] = result["body"]
                                                st.success("✅ Email generated from templates!")
                                                st.rerun()
                                            else:
                                                st.error(f"Failed to generate email: {result['message']}")
                                        else:
                                            st.warning("No templates found for this persona and stage")
                                    except Exception as e:
                                        st.error(f"Error using template: {str(e)}")
                        
                        with col2:
                            # Add checkbox for template option
                            # save_as_template = st.checkbox("Save as template", key=f"template_checkbox_{idx}", 
                            #                               help="Check this to save as a reusable template")
                            save_as_template = True
                            if st.button("💾 Save Email", key=f"save_{idx}"):
                                with st.spinner("Saving email..."):
                                    try:
                                        template_manager = SimpleEmailManager()
                                        # Save the draft first
                                        draft_result = asyncio.run(template_manager.save_draft(
                                            lead_id=lead.get('linkedin_url', ''),  # Using LinkedIn URL as lead_id
                                            subject=st.session_state.get(f"subject_{idx}", ""),
                                            body=st.session_state.get(f"body_{idx}", ""),
                                            persona=persona,
                                            stage=stage
                                        ))
                                        
                                        if draft_result["status"] == "success":
                                            # If checkbox is checked, mark as template
                                            if save_as_template:
                                                template_result = asyncio.run(template_manager.mark_as_template(
                                                    draft_id=draft_result["data"]["id"],
                                                    persona=persona,
                                                    stage=stage
                                                ))
                                                
                                                if template_result["status"] == "success":
                                                    st.success("✅ Saved as template for future use!")
                                                else:
                                                    st.error(f"Failed to mark as template: {template_result['message']}")
                                            else:
                                                st.success("✅ Email saved!")
                                        else:
                                            st.error(f"Failed to save email: {draft_result['message']}")
                                    except Exception as e:
                                        st.error(f"Error saving email: {str(e)}")
                        
                        # Show drafts if available
                        # drafts = asyncio.run(SimpleEmailManager().get_drafts(lead.get('linkedin_url', '')))
                        # if drafts:
                        #     with st.expander("📋 Saved Emails"):
                        #         for draft in drafts:
                        #             col1, col2 = st.columns([3, 1])
                        #             with col1:
                        #                 st.markdown(f"**Subject:** {draft['subject']}")
                        #                 st.markdown(f"**Status:** {draft['status']}")
                        #                 if draft.get('persona') and draft.get('stage'):
                        #                     st.markdown(f"**For:** {draft['persona']} / {draft['stage']}")
                                    
                        #             with col2:
                        #                 if st.button("Use This Email", key=f"use_draft_{draft['id']}"):
                        #                     st.session_state[f"subject_{idx}"] = draft['subject']
                        #                     st.session_state[f"body_{idx}"] = draft['body']
                        #                     st.rerun()
                                    
                        #             st.markdown("---")
                        
                        # Email composition fields
                        subject = st.text_input(
                            "Subject", 
                            value=st.session_state.get(f"subject_{idx}", f"Quick question about {lead.get('companyName', 'your company')}"), 
                            key=f"subject_{idx}"
                        )
                        
                        st.markdown("""
                        **Email Body Tips:**
                        1. Write your email in normal text format
                        2. For links, use: [text](url) - e.g., [our website](https://example.com)
                        3. Use normal paragraphs and line breaks
                        """)
                        
                        # Email template
                        default_template = f"""Hi {lead.get('firstName', '')},

I noticed your role as {lead.get('jobTitle', '')} at {lead.get('companyName', '')}. 

[Your personalized message here]

Best regards,
[Your name]"""
                        
                        email_body = st.text_area(
                            "Body", 
                            value=st.session_state.get(f"body_{idx}", default_template),
                            height=200, 
                            key=f"body_{idx}"
                        )

                        # Add send email button
                        if st.button("📤 Send Email", key=f"send_{idx}"):
                            with st.spinner("Sending email..."):
                                # Send the email
                                result = email_manager.send_email(
                                    lead.get("email", ""),
                                    subject,
                                    email_body
                                )
                                
                                if result["status"] == "success":
                                    st.success("✅ Email sent successfully!")
                                    # Mark as sent in session state
                                    st.session_state[email_key] = True
                                    st.session_state[compose_key] = False
                                    st.rerun()
                                else:
                                    st.error(f"Failed to send email: {result['message']}")

def icp_configuration_page():
    """ICP Configuration page"""
    st.header("⚙️ ICP Configuration")
    
    # Create tabs for different configurations
    icp_tab, email_tab = st.tabs(["🎯 ICP Scoring Prompt", "📧 Email Prompt"])
    
    with icp_tab:
        st.markdown("### ICP Scoring Prompt Configuration")
        st.markdown("Customize the AI prompt used for scoring leads against your ICP criteria")
        
        # Get the default prompt from AIICPScorer class
        icp_scorer = AIICPScorer(OPENAI_API_KEY)
        default_icp_prompt = icp_scorer.default_icp_prompt
        
        # ICP prompt configuration
        icp_prompt = st.text_area(
            "ICP Scoring Prompt",
            value=st.session_state.get("icp_prompt", default_icp_prompt),
            height=600,
            help="Customize the prompt used by AI to score leads against your ICP criteria. Use {profile_json} as a placeholder for profile data."
        )
        
        # Show preview of prompt structure
        with st.expander("📝 ICP Prompt Structure Guide"):
            st.markdown("""
            ### Prompt Structure Requirements:
            1. Keep the `{profile_json}` placeholder - it's used to inject profile data
            2. Define your ICP criteria clearly (industries, roles, company sizes)
            3. Specify the exact JSON response format required
            4. Include scoring guidelines (0-10 scale)
            5. Maintain the validation rules for the response
            
            ### Required JSON Response Fields:
            ```json
            {
                "industry_fit": <score 0-10>,
                "role_fit": <score 0-10>,
                "company_size_fit": <score 0-10>,
                "decision_maker": <score 0-10>,
                "total_score": <weighted average 0-10>,
                "icp_category": "<operations|field_service|none>",
                "reasoning": "<brief explanation>"
            }
            ```
            """)
    
    with email_tab:
        st.markdown("### Email Prompt Configuration")
        st.markdown("Customize the AI prompt used for generating cold outreach emails")
        
        # Get the default prompt from EmailGenerator class
        email_generator = EmailGenerator(OPENAI_API_KEY)
        default_email_prompt = email_generator.default_email_prompt
        
        # Email prompt configuration
        email_prompt = st.text_area(
            "Email Generation Prompt",
            value=st.session_state.get("email_prompt", default_email_prompt),
            height=400,
            help="Customize the prompt used by AI to generate cold outreach emails. Use {lead_info} as a placeholder for lead information."
        )
        
        # Show preview of prompt structure
        with st.expander("📝 Email Prompt Structure Guide"):
            st.markdown("""
            ### Prompt Structure Tips:
            1. Keep the `{lead_info}` placeholder - it's used to inject lead information
            2. Include clear guidelines for tone and style
            3. Specify any industry-specific context
            4. Define email structure preferences
            5. List any phrases or approaches to avoid
            """)
    
    if st.button("💾 Save Configuration"):
        try:
            # Validate that the prompts contain their required placeholders
            if "{profile_json}" not in icp_prompt:
                st.error("ICP Scoring prompt must contain the {profile_json} placeholder!")
                return
                
            if "{lead_info}" not in email_prompt:
                st.error("Email prompt must contain the {lead_info} placeholder!")
                return
            
            # Save prompts to session state
            st.session_state["icp_prompt"] = icp_prompt
            st.session_state["email_prompt"] = email_prompt
            
            st.success("✅ Configuration saved successfully!")
            st.info("Note: In a production environment, this would be saved to a persistent storage.")
            
        except Exception as e:
            st.error(f"Error saving configuration: {str(e)}")

def email_dashboard_page():
    """Email dashboard page showing email stats and events"""
    st.title("📊 Email Dashboard")
    st.markdown("Track your email campaign performance and engagement metrics")
    
    # Initialize Supabase client with service role key
    try:
        supabase = create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_SERVICE_ROLE_KEY"]  # Use service role key instead of anon key
        )
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {str(e)}")
        return
    
    # Create columns for key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # Get email events from Supabase with better error handling
        try:
            response = supabase.table("email_events").select("*").execute()
            if not response.data:
                st.info("No email events found in the database.")
                return
            events = response.data
        except Exception as fetch_error:
            st.error(f"Error fetching email events: {str(fetch_error)}")
            return
        
        # Convert events to DataFrame for easier manipulation
        df = pd.DataFrame(events)
        
        # Calculate metrics
        total_sent = len(df[df["event_type"] == "email.sent"])
        total_delivered = len(df[df["event_type"] == "email.delivered"])
        total_opened = len(df[df["event_type"] == "email.opened"])
        total_clicked = len(df[df["event_type"] == "email.clicked"])
        total_bounced = len(df[df["event_type"] == "email.bounced"])
        total_complained = len(df[df["event_type"] == "email.complained"])
        
        # Calculate rates with safe division
        delivery_rate = (total_delivered / total_sent * 100) if total_sent > 0 else 0
        open_rate = (total_opened / total_delivered * 100) if total_delivered > 0 else 0
        click_rate = (total_clicked / total_opened * 100) if total_opened > 0 else 0
        bounce_rate = (total_bounced / total_sent * 100) if total_sent > 0 else 0
        
        with col1:
            st.metric("Total Sent", total_sent)
            st.metric("Delivery Rate", f"{delivery_rate:.1f}%")
            
        with col2:
            st.metric("Total Opened", total_opened)
            st.metric("Open Rate", f"{open_rate:.1f}%")
            
        with col3:
            st.metric("Total Clicked", total_clicked)
            st.metric("Click Rate", f"{click_rate:.1f}%")
            
        with col4:
            st.metric("Bounces", total_bounced)
            st.metric("Bounce Rate", f"{bounce_rate:.1f}%")
        
        # Event timeline
        st.subheader("📈 Event Timeline")
        
        # Convert created_at to datetime and sort
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at", ascending=False)
        
        # Display recent events with better column selection and formatting
        display_columns = ["created_at", "event_type", "email_id", "to_email", "subject"]
        display_df = df[display_columns].copy()
        
        # Format datetime for display
        display_df["created_at"] = display_df["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True
        )
        
        # Event type distribution
        st.subheader("📊 Event Distribution")
        event_counts = df["event_type"].value_counts()
        
        # Use plotly for better interactive charts
        import plotly.express as px
        
        fig = px.bar(
            x=event_counts.index,
            y=event_counts.values,
            labels={"x": "Event Type", "y": "Count"},
            title="Email Event Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed event analysis
        st.subheader("🔍 Detailed Analysis")
        
        # Show bounce analysis if there are bounces
        bounces_df = df[df["event_type"] == "email.bounced"]
        if not bounces_df.empty:
            st.markdown("### 🚫 Bounce Analysis")
            
            # Group bounces by type
            bounce_types = bounces_df["bounce_type"].value_counts()
            
            fig_bounces = px.bar(
                x=bounce_types.index,
                y=bounce_types.values,
                labels={"x": "Bounce Type", "y": "Count"},
                title="Email Bounce Types"
            )
            st.plotly_chart(fig_bounces, use_container_width=True)
            
            # Show bounce messages
            st.markdown("#### Recent Bounce Messages")
            bounce_messages = bounces_df[["created_at", "bounce_type", "bounce_message", "to_email"]].head(5)
            bounce_messages["created_at"] = bounce_messages["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(bounce_messages, hide_index=True, use_container_width=True)
        
        # Show click analysis if there are clicks
        clicks_df = df[df["event_type"] == "email.clicked"]
        if not clicks_df.empty:
            st.markdown("### 🖱️ Click Analysis")
            
            # Group clicks by link
            link_clicks = clicks_df["click_link"].value_counts()
            
            fig_clicks = px.bar(
                x=link_clicks.index,
                y=link_clicks.values,
                labels={"x": "Link", "y": "Clicks"},
                title="Most Clicked Links"
            )
            st.plotly_chart(fig_clicks, use_container_width=True)
            
            # Show click details
            st.markdown("#### Recent Click Details")
            click_details = clicks_df[["created_at", "click_link", "click_user_agent", "to_email"]].head(5)
            click_details["created_at"] = click_details["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(click_details, hide_index=True, use_container_width=True)
        
        # Time-based analysis
        st.subheader("📅 Time-Based Analysis")
        
        # Group events by date
        df["date"] = df["created_at"].dt.date
        daily_events = df.groupby(["date", "event_type"]).size().unstack(fill_value=0)
        
        fig_timeline = px.line(
            daily_events,
            labels={"date": "Date", "value": "Count", "event_type": "Event Type"},
            title="Email Events Over Time"
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error processing email events: {str(e)}")
        st.markdown("### 🔍 Debug Information")
        st.code(f"Error details: {str(e)}")
        
        # Check Supabase connection
        try:
            test_response = supabase.table("email_events").select("count").limit(1).execute()
            st.success("✅ Supabase connection is working")
        except Exception as conn_error:
            st.error(f"❌ Supabase connection test failed: {str(conn_error)}")

def create_lead_agent():
    """Create the lead generation agent"""
    
    sheets_service = setup_google_sheets()
    if not sheets_service:
        return None
    
    if not APIFY_API_TOKENS:
        st.error("No Apify API tokens configured. Please add at least one token to secrets.")
        return None
    
    scraping_tool = LeadScrapingTool(APIFY_API_TOKENS, sheets_service, GOOGLE_API_KEY, GOOGLE_CSE_ID)
    
    def scrape_leads_with_config(query_json: str, method: str = "apollo") -> str:
        """Wrapper function to include leads_per_query from session state"""
        leads_per_query = st.session_state.get("leads_per_query", 20)
        return scraping_tool.scrape_leads(query_json, method, num_results=leads_per_query)
    
    tools = [
        Tool(
            name="leadScraping",
            description="Use this tool to scrape leads into a Google Sheet. Only call this tool once you have enough information to complete the desired JSON search query.",
            func=scrape_leads_with_config
        )
    ]
    
    prompt = PromptTemplate.from_template("""You are Lead Generation Joe, a lead scraping assistant.

STRICT FORMAT RULES:
1. ALWAYS start with "Thought:"
2. NEVER skip the Thought step
3. NEVER use Action: None
4. Use EXACTLY this format:

When missing information:
Thought: I need to ask for specific missing information
Final Answer: Enter all three pieces of information together.

When you have all information:
Thought: I have location, business, and job title information
Action: leadScraping
Action Input: [{{"location": ["city+country"], "business": ["type"], "job_title": ["title"]}}]
Observation: <wait for result>
Final Answer: <summarize result>

Example correct responses:
---
Thought: I don't have any information yet
Final Answer: Hi! I'm Lead Generation Joe. Please provide the locations (e.g., "New York United States"), business types (e.g., "Manufacturing"), and job titles (e.g., "Plant Manager") you want to search for.
---
Thought: I have all required information
Action: leadScraping from {tool_names}
Action Input: [{{"location":["new+york+united+states"],"business":["manufacturing"],"job_title":["plant+manager"]}}]
--- 

Available tools: {tools}

Question: {input}
Thought: {agent_scratchpad}""")
    
    llm = ChatOpenAI(
        model="openai/o3-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        base_url="https://openrouter.ai/api/v1/",
    )
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    
    return agent_executor

def main():
    st.set_page_config(
        page_title="Lead Generation System",
        page_icon="🎯",
        layout="wide"
    )
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🎯 Lead Generation", "🗃️ Leads Database", "📝 Google sheets", "📊 Email Dashboard", "⚙️ ICP Configuration"]
    )
    
    if page == "🎯 Lead Generation":
        lead_generation_page()
    elif page == "🗃️ Leads Database":
        leads_database_page()
    elif page == "📝 Google sheets":
        email_management_page()
    elif page == "📊 Email Dashboard":
        email_dashboard_page()
    elif page == "⚙️ ICP Configuration":
        icp_configuration_page()

def lead_generation_page():
    """Lead generation page with chat interface and direct query option"""
    st.title("🎯 Lead Generation System")
    st.markdown("Powered by AI Agent + Web Scraping + Data Enrichment + ICP Scoring")
    
    # Lead Generation content (removed query history tab)
    
    # Add a toggle for input method
    input_method = st.radio(
        "Choose input method:",
        [ "Direct Query Form","Chat with Lead Generation Joe"],
        horizontal=True
    )

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Leads per query selector
        st.subheader("🎯 Query Settings")
        leads_per_query = st.selectbox(
            "Leads per search query",
            options=[1, 5, 10, 20, 25, 50, 100],
            index=3,  # Default to 20
            help="Number of LinkedIn profiles to fetch per search query. More leads = longer processing time."
        )
        st.session_state["leads_per_query"] = leads_per_query
    
        # API Key checks
        st.markdown("---")
        st.subheader("🔑 API Status")
        api_status = {
            "OpenAI API": bool(OPENAI_API_KEY),
            "Apollo.io API": bool(APIFY_API_TOKENS),
            "Google Search API": bool(GOOGLE_API_KEY and GOOGLE_CSE_ID),
            "Google Sheets": bool(GOOGLE_SHEETS_CREDENTIALS),
            "Resend API": bool(RESEND_API_KEY)
        }
        
        # Show number of Apify tokens if available
        if APIFY_API_TOKENS:
            st.info(f"📊 {len(APIFY_API_TOKENS)} Apify API token(s) configured")
        
        for service, status in api_status.items():
            if status:
                st.success(f"✅ {service}")
            else:
                st.error(f"❌ {service} - Please add to secrets")
    
        st.markdown("---")
        st.markdown("### How to use:")
        st.markdown("**Chat Interface:**")
        st.markdown("1. Start by saying 'Hi' to Lead Generation Joe")
        st.markdown("2. Provide locations, businesses, and job titles")
        st.markdown("3. Joe will automatically process and save leads")
        st.markdown("")
        st.markdown("**Direct Query Form:**")
        st.markdown("1. Choose between Apollo.io or Google Search method")
        st.markdown("2. Fill in job titles, locations, and industries")
        st.markdown("3. System will automatically:")
        st.markdown("   - 🔍 Search for LinkedIn profiles")
        st.markdown("   - 🤖 Enrich profiles with data")
        st.markdown("   - 🎯 Score leads based on ICP criteria")
        st.markdown("   - 💾 Save all data to Google Sheets")
        st.markdown("4. Use the Email Management page to send cold emails")
        
        st.markdown("---")
        st.markdown("### ICP Scoring")
        st.markdown("Leads are automatically scored based on:")
        st.markdown("- Job Title (0-10 points)")
        st.markdown("- Company Size (0-10 points)")
        st.markdown("- Industry (0-10 points)")
        st.markdown("- Location (0-10 points)")
        st.markdown("- **Total: 0-40 points (converted to %)**")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = create_lead_agent()

    
    # Chat-based interface
    if input_method == "Chat with Lead Generation Joe":
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Chat with Lead Generation Joe..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get agent response
            with st.chat_message("assistant"):
                if st.session_state.agent:
                    with st.spinner("Lead Generation Joe is thinking..."):
                        try:
                            response = st.session_state.agent.invoke({"input": prompt})
                            st.markdown(response["output"])
                            st.session_state.messages.append({"role": "assistant", "content": response["output"]})
                        except Exception as e:
                            error_msg = f"Sorry, I encountered an error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    error_msg = "Agent initialization failed. Please check your API configurations."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Direct query form interface
    else:
        sheets_service = setup_google_sheets()
        if not sheets_service:
            st.error("Unable to connect to Google Sheets for direct queries. Please check your API configuration.")
            return
        
        if not APIFY_API_TOKENS:
            st.error("No Apify API tokens configured. Please add at least one token to secrets.")
            return
        
        st.subheader("🔍 Direct Lead Search")
        st.markdown("""
        Specify your search criteria to generate leads using Apollo.io or Google Search + Apify enrichment:
        
        1. Choose your lead generation method
        2. Select job titles, locations, and industries from the dropdown menus
        3. Add custom values if needed
        4. Choose company size ranges
        5. Click 'Generate Leads' to start the search
        
        For best results, use specific locations and job titles.
        """)
        
        with st.form("direct_lead_search_form"):
            # Method Selection
            st.subheader("🔧 Lead Generation Method")
            method = st.radio(
                "Choose your lead generation method:",
                ["Google Search + Apify Enrichment","Apollo.io"],
                help="Apollo.io: Fast, structured data from Apollo's database. Google Search: Custom search with Apify enrichment."
            )
            
            # Job Title - Single select with common options and custom input
            st.subheader("👔 Job Title (Select One)")
            preset_title = st.selectbox(
                "Select a job title:",
                [
                    "-- Select from list --",
                    "Operations Head", "Operations Manager", "Plant Manager", "Production Engineer",
                    "Facility Manager", "Service Head", "Asset Manager",
                    "Maintenance Manager", "Operations Director", "COO"
                ]
            )
            
            custom_title = st.text_input("Or enter custom job title:", "")
            
            # Location - Single select with common options and custom input
            st.subheader("📍 Location (Select One)")
            preset_location = st.selectbox(
                "Select a location:",
                [
                    "-- Select from list --",
                    "United States", "Canada", "United Kingdom", "Australia", "Singapore", "India"
                ]
            )
            
            custom_location = st.text_input("Or enter custom location:", "")
            
            # Industry - Single select with common options and custom input
            st.subheader("🏭 Industry (Select One)")
            preset_industry = st.selectbox(
                "Select an industry:",
                [
                    "-- Select from list --",
                    "Manufacturing", "Industrial Automation", "Consumer Electronics"
                ]
            )
            
            custom_industry = st.text_input("Or enter custom industry:", "")
            
            # Company Size - Only for Apollo method
            company_sizes = []
            try:
                st.info("ℹ️ Company size filtering is not available for Google Search method.")
                st.subheader("🏢 Company Size (Apollo Only)")
                company_sizes = st.multiselect(
                    "Select company size ranges:",
                    [
                        "1,10", "11,20", "21,50", "51,100", "101,200", "201,500", "501,1000"
                    ],
                    default=["1,10", "11,20", "21,50", "51,100"]
                )
            except:
                pass
            
            # Submit button
            submit_button = st.form_submit_button("🚀 Generate Leads")
        
        if submit_button:
            # Create query from selections - single values only
            job_title = custom_title if custom_title else (preset_title if preset_title != "-- Select from list --" else "")
            location = custom_location if custom_location else (preset_location if preset_location != "-- Select from list --" else "")
            industry = custom_industry if custom_industry else (preset_industry if preset_industry != "-- Select from list --" else "")
            
            # Check if we have enough info to proceed
            if not job_title or not location or not industry:
                st.error("Please provide a job title, location, and industry.")
                return
            
            # Format for the query - single values in lists for compatibility
            formatted_job_titles = [job_title.replace(" ", "+")]
            formatted_locations = [location.replace(" ", "+")]
            formatted_industries = [industry.replace(" ", "+")]
            
            # Ensure company sizes are in the correct format (already in comma format)
            formatted_company_sizes = company_sizes
            
            # Create query JSON from form data
            query_json = json.dumps({
                "query": [{
                    "job_title": formatted_job_titles,
                    "location": formatted_locations,
                    "business": formatted_industries,
                    "employee_ranges": formatted_company_sizes
                }]
            })
            method_param = "apollo" if method == "Apollo.io" else "google_search"
            
            # Create a scraping tool instance
            scraping_tool = LeadScrapingTool(APIFY_API_TOKENS, sheets_service, GOOGLE_API_KEY, GOOGLE_CSE_ID)
            
            # Check if required credentials are available for the selected method
            if method_param == "google_search" and (not GOOGLE_API_KEY or not GOOGLE_CSE_ID):
                st.error("❌ Google Search method requires Google API Key and Custom Search Engine ID. Please add them to your secrets.")
                return
            
            # Execute the scraping process
            method_display = "Apollo.io" if method_param == "apollo" else "Google Search + Apify"
            with st.spinner(f"🔍 Generating leads using {method_display}..."):
                try:
                    # Store generation timestamp in session state
                    generation_timestamp = datetime.now().isoformat()
                    st.session_state['last_generation_timestamp'] = generation_timestamp
                    
                    result = scraping_tool.scrape_leads(query_json, method=method_param, num_results=leads_per_query)
                    st.success("✅ Lead generation completed!")
                    st.markdown(result)
                    
                    # Display leads immediately after generation
                    st.markdown("---")
                    display_generated_leads()
                    
                except Exception as e:
                    st.error(f"❌ Error generating leads: {str(e)}")

def get_supabase_client():
    """Initialize and return Supabase client."""
    try:
        from supabase import create_client
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Error initializing Supabase client: {str(e)}")
        return None

def display_generated_leads():
    """Display newly generated leads from current session in table format with email management."""
    
    # Email modal function for generated leads
    @st.dialog("📧 Email Management")
    def email_modal(lead_name, lead_title, lead_company, lead_email, lead_first_name, lead_data):
        st.markdown(f"### 📧 Email for {lead_name}")
        
        # Lead information display
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Lead:** {lead_name} ({lead_title})")
        with col2:
            st.markdown(f"**Company:** {lead_company}")
        
        # Email status tracking
        email_key = f"email_sent_{lead_data.get('id', '')}"
        
        if st.session_state.get(email_key, False):
            st.success("✅ Email already sent to this lead")
            if st.button("📧 Send Another Email", key="resend_email"):
                st.session_state[email_key] = False
                st.rerun()
        
        # Email composition form
        st.markdown("#### ✍️ Compose Email")
        
        # Template selection
        template_col1, template_col2 = st.columns([1, 1])
        with template_col1:
            persona = st.selectbox(
                "Select Persona",
                ["operations_manager", "facility_manager", "maintenance_manager", "plant_manager"],
                key="email_persona"
            )
        with template_col2:
            stage = st.selectbox(
                "Email Stage",
                ["initial_outreach", "follow_up", "meeting_request"],
                key="email_stage"
            )
        
        # Template and save buttons
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button("🔄 Use Template", key="use_template"):
                with st.spinner("Generating from template..."):
                    try:
                        template_manager = SimpleEmailManager()
                        templates = asyncio.run(template_manager.retrieve_templates(persona, stage))
                        if templates:
                            # Convert lead data to expected format
                            lead_data_formatted = {
                                'fullName': lead_name,
                                'firstName': lead_first_name,
                                'jobTitle': lead_title,
                                'companyName': lead_company,
                                'email': lead_email
                            }
                            result = asyncio.run(template_manager.generate_email(lead_data_formatted, templates))
                            if result["status"] == "success":
                                st.session_state["email_subject"] = result["subject"]
                                st.session_state["email_body"] = result["body"]
                                st.success("✅ Email generated from templates!")
                                st.rerun()
                            else:
                                st.error(f"Failed to generate email: {result['message']}")
                        else:
                            st.warning("No templates found for this persona and stage")
                    except Exception as e:
                        st.error(f"Error using template: {str(e)}")
        
        with btn_col2:
            if st.button("💾 Save as Template", key="save_template"):
                with st.spinner("Saving email as template..."):
                    try:
                        template_manager = SimpleEmailManager()
                        draft_result = asyncio.run(template_manager.save_draft(
                            lead_id=lead_data.get('id', ''),
                            subject=st.session_state.get("email_subject", ""),
                            body=st.session_state.get("email_body", ""),
                            persona=persona,
                            stage=stage
                        ))
                        
                        if draft_result["status"] == "success":
                            # Mark as template
                            template_result = asyncio.run(template_manager.mark_as_template(
                                draft_id=draft_result["data"]["id"],
                                persona=persona,
                                stage=stage
                            ))
                            
                            if template_result["status"] == "success":
                                st.success("✅ Saved as template for future use!")
                            else:
                                st.error(f"Failed to mark as template: {template_result['message']}")
                        else:
                            st.error(f"Failed to save email: {draft_result['message']}")
                    except Exception as e:
                        st.error(f"Error saving email: {str(e)}")
        
        # Email composition fields
        subject = st.text_input(
            "Subject", 
            value=st.session_state.get("email_subject", f"Quick question about {lead_company}"),
            key="email_subject"
        )
        
        # Default email template
        default_template = f"""Hi {lead_first_name},

I noticed your role as {lead_title} at {lead_company}.

[Your personalized message here]

Best regards,
[Your name]"""
        
        email_body = st.text_area(
            "Body", 
            value=st.session_state.get("email_body", default_template),
            height=200,
            key="email_body"
        )
        
        # Send and close buttons
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button("📤 Send Email", key="send_email", type="primary"):
                with st.spinner("Sending email..."):
                    email_manager = EmailManager(RESEND_API_KEY, "resend.dev", SENDER_EMAIL)
                    result = email_manager.send_email(
                        lead_email,
                        subject,
                        email_body
                    )
                    
                    if result["status"] == "success":
                        st.success("✅ Email sent successfully!")
                        # Mark as sent in session state
                        st.session_state[email_key] = True
                        
                        # Update lead status in database
                        try:
                            supabase_client = get_supabase_client()
                            supabase_client.table("leads").update({
                                "send_email_status": "Sent"
                            }).eq("id", lead_data.get('id')).execute()
                        except Exception as e:
                            st.warning(f"Could not update email status in database: {str(e)}")
                        
                        # Clear email content from session state
                        if "email_subject" in st.session_state:
                            del st.session_state["email_subject"]
                        if "email_body" in st.session_state:
                            del st.session_state["email_body"]
                        
                        st.rerun()
                    else:
                        st.error(f"Failed to send email: {result['message']}")
        
        with btn_col2:
            if st.button("❌ Close", key="close_email"):
                # Clear email content from session state
                if "email_subject" in st.session_state:
                    del st.session_state["email_subject"]
                if "email_body" in st.session_state:
                    del st.session_state["email_body"]
                st.rerun()
    
    st.subheader("📊 Generated Leads")
    st.markdown("View and manage your newly generated leads")
    
    # Check if there's a generation timestamp in session state
    if 'last_generation_timestamp' not in st.session_state:
        st.info("No leads generated in this session yet. Generate some leads first!")
        return
    
    # Initialize Supabase client
    try:
        supabase_client = get_supabase_client()
        if not supabase_client:
            st.error("❌ Supabase client not initialized. Cannot display leads.")
            return
    except Exception as e:
        st.error(f"❌ Error connecting to Supabase: {str(e)}")
        return
    
    # Initialize email manager
    email_manager = EmailManager(RESEND_API_KEY, "resend.dev", SENDER_EMAIL)
    
    try:
        # Get leads created after the last generation timestamp
        generation_timestamp = st.session_state['last_generation_timestamp']
        response = supabase_client.table("leads").select("*").gte("created_at", generation_timestamp).order("created_at", desc=True).execute()
        
        if not response.data:
            st.info("No newly generated leads found. Generate some leads first!")
            return
        
        leads_data = response.data
        total_leads = len(leads_data)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Generated Leads", total_leads)
        with col2:
            # Count leads with emails
            leads_with_email = sum(1 for lead in leads_data if lead.get('email') or lead.get('email_address'))
            st.metric("Leads with Email", leads_with_email)
        with col3:
            # Average ICP score
            icp_scores = [float(lead.get('icp_percentage', 0)) for lead in leads_data if lead.get('icp_percentage')]
            avg_icp = sum(icp_scores) / len(icp_scores) if icp_scores else 0
            st.metric("Avg ICP Score", f"{avg_icp:.1f}%")
        
        st.markdown("---")
        
        # Convert to DataFrame for easier manipulation
        leads_df = pd.DataFrame(leads_data)
        
        # Display all leads without filters
        filtered_df = leads_df.copy()
        
        # Display results count
        st.info(f"📊 Showing {len(filtered_df)} generated leads")
        
        if filtered_df.empty:
            st.warning("No leads found.")
            return
        
        # Display leads in table format
        st.subheader("📋 Generated Leads Table")
        st.info("💡 To manage and filter your generated leads, go to the **Leads Database** page")
        
        # Table headers
        # header_cols = st.columns([2.5, 2, 2, 1.5, 1.5, 1, 1.5])
        header_cols = st.columns([2.5, 2, 2, 1.5, 1.5, 1])
        with header_cols[0]:
            st.markdown("**👤 Name & Title**")
        with header_cols[1]:
            st.markdown("**🏢 Company**")
        with header_cols[2]:
            st.markdown("**📧 Email**")
        with header_cols[3]:
            st.markdown("**📍 Location**")
        with header_cols[4]:
            st.markdown("**📱 Social Media**")
        with header_cols[5]:
            st.markdown("**🎯 ICP**")
        # with header_cols[6]:
        #     st.markdown("**⚡ Actions**")
        
        st.markdown("---")
        
        # Display each lead as a table row
        for lead_idx, lead in filtered_df.iterrows():
            # Handle different field naming conventions
            name = lead.get('full_name') or lead.get('fullName', 'Unknown')
            first_name = lead.get('first_name') or lead.get('firstName', name.split()[0] if name != 'Unknown' else '')
            title = lead.get('job_title') or lead.get('jobTitle', 'Unknown Title')
            company = lead.get('company_name') or lead.get('companyName', 'Unknown Company')
            industry = lead.get('company_industry') or lead.get('companyIndustry', 'Unknown Industry')
            linkedin_url = lead.get('linkedin_url', '')
            email = lead.get('email') or lead.get('email_address', 'No email')
            
            # Location
            location_parts = []
            if lead.get('city'):
                location_parts.append(str(lead.get('city')))
            if lead.get('state'):
                location_parts.append(str(lead.get('state')))
            if lead.get('country'):
                location_parts.append(str(lead.get('country')))
            location = ', '.join(location_parts) if location_parts else 'N/A'
            
            # ICP Score
            icp_score = lead.get('icp_percentage') or lead.get('icp_score', 0)
            icp_grade = lead.get('icp_grade', 'N/A')
            
            # Create table row
            row_cols = st.columns([2.5, 2, 2, 1.5, 1.5, 1, 1.5])
            
            with row_cols[0]:
                # Create name with LinkedIn link
                if linkedin_url:
                    st.markdown(f"**[{name}]({linkedin_url})**")
                else:
                    st.markdown(f"**{name}**")
                st.caption(title)
            
            with row_cols[1]:
                # Create company with LinkedIn and website links
                company_links = []
                
                # Add company LinkedIn URL if available
                company_linkedin = lead.get('company_linkedin') or lead.get('companyLinkedin')
                if company_linkedin:
                    if not company_linkedin.startswith(('http://', 'https://')):
                        company_linkedin = f"https://{company_linkedin}"
                    company_links.append(f"[{company}]({company_linkedin})")
                else:
                    company_links.append(company)
                
                st.markdown(" ".join(company_links))
                
                # Show website as caption if available
                company_website = lead.get('company_website') or lead.get('companyWebsite', '')
                if company_website:
                    if not company_website.startswith(('http://', 'https://')):
                        company_website = f"https://{company_website}"
                    st.caption(f"🌐 [Website]({company_website})")
                else:
                    st.caption(industry)
            
            with row_cols[2]:
                if email and email != 'No email':
                    st.markdown(f"📧 {email}")
                else:
                    st.write("N/A")
            
            with row_cols[3]:
                st.write(location)
            
            with row_cols[4]:
                # Social Media links
                social_links = []
                twitter = lead.get('company_twitter', '')
                facebook = lead.get('company_facebook', '')
                
                if twitter:
                    if not twitter.startswith(('http://', 'https://')):
                        twitter = f"https://{twitter}"
                    social_links.append(f"[🐦]({twitter})")
                if facebook:
                    if not facebook.startswith(('http://', 'https://')):
                        facebook = f"https://{facebook}"
                    social_links.append(f"[📘]({facebook})")
                
                if social_links:
                    st.markdown(" ".join(social_links))
                else:
                    st.write("N/A")
            
            with row_cols[5]:
                if icp_score and float(icp_score) > 0:
                    score = float(icp_score)
                    if score >= 80:
                        st.markdown(f"🟢 **{score:.0f}%**")
                    elif score >= 60:
                        st.markdown(f"🟡 **{score:.0f}%**")
                    else:
                        st.markdown(f"🔴 **{score:.0f}%**")
                    st.caption(f"Grade: {icp_grade}")
                else:
                    st.write("N/A")
            
            with row_cols[6]:
                # Email management button
                if email and email != 'No email':
                    email_key = f"email_modal_{lead.get('id', lead_idx)}"
                    if st.button("📧 Email", key=f"email_btn_{lead_idx}", help="Manage email for this lead"):
                        st.session_state[email_key] = True
                        st.session_state[f"current_lead_name_{lead_idx}"] = name
                        st.session_state[f"current_lead_title_{lead_idx}"] = title
                        st.session_state[f"current_lead_company_{lead_idx}"] = company
                        st.session_state[f"current_lead_email_{lead_idx}"] = email
                        st.session_state[f"current_lead_first_name_{lead_idx}"] = first_name
                        st.session_state[f"current_lead_data_{lead_idx}"] = lead
                else:
                    st.caption("No email")
            
            # Add separator between rows
            if lead_idx < len(filtered_df) - 1:
                st.markdown("<hr style='margin: 0.5rem 0; border: 1px solid #e0e0e0;'>", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error loading leads: {str(e)}")
    
    # Handle email modal triggers for generated leads
    for lead_idx in range(len(filtered_df) if 'filtered_df' in locals() and not filtered_df.empty else 0):
        email_key = f"email_modal_{filtered_df.iloc[lead_idx].get('id', lead_idx)}"
        if st.session_state.get(email_key, False):
            lead_name = st.session_state.get(f"current_lead_name_{lead_idx}", "Unknown")
            lead_title = st.session_state.get(f"current_lead_title_{lead_idx}", "Unknown Title")
            lead_company = st.session_state.get(f"current_lead_company_{lead_idx}", "Unknown Company")
            lead_email = st.session_state.get(f"current_lead_email_{lead_idx}", "")
            lead_first_name = st.session_state.get(f"current_lead_first_name_{lead_idx}", "")
            lead_data = st.session_state.get(f"current_lead_data_{lead_idx}", {})
            
            email_modal(lead_name, lead_title, lead_company, lead_email, lead_first_name, lead_data)
            st.session_state[email_key] = False


def leads_database_page():
    """Display all scraped leads from Supabase in table format with email management."""
    
    # Define email modal function outside the loop to avoid dialog conflicts
    @st.dialog("📧 Email Management")
    def email_modal(name, title, company, email, first_name, lead, lead_idx, email_key):
        # Display lead info in modal
        st.markdown(f"**Lead:** {name} ({title})")
        st.markdown(f"**Company:** {company}")
        st.markdown(f"**Email:** {email}")
        st.markdown("---")
        
        # Initialize email manager
        email_manager = EmailManager(RESEND_API_KEY, "resend.dev", SENDER_EMAIL)
        
        # Email status tracking
        sent_key = f"email_sent_{lead.get('id', lead_idx)}"
        compose_key = f"compose_{sent_key}"
        
        if st.session_state.get(sent_key, False):
            st.success("✅ Email already sent to this lead")
            if st.button("📧 Send Another Email", key=f"resend_{lead_idx}"):
                st.session_state[sent_key] = False
                st.session_state[compose_key] = False
                st.rerun()
        else:
            # Email composition form (show directly)
            st.markdown("#### 📨 Compose Email")
            
            # Template selection
            template_col1, template_col2 = st.columns([1, 1])
            with template_col1:
                persona = st.selectbox(
                    "Select Persona",
                    ["operations_manager", "facility_manager", "maintenance_manager", "plant_manager"],
                    key=f"persona_{lead_idx}"
                )
            with template_col2:
                stage = st.selectbox(
                    "Email Stage",
                    ["initial_outreach", "follow_up", "meeting_request"],
                    key=f"stage_{lead_idx}"
                )
            
            # Template and save buttons
            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                if st.button("🔄 Use Template", key=f"template_{lead_idx}"):
                    with st.spinner("Generating from template..."):
                        try:
                            template_manager = SimpleEmailManager()
                            templates = asyncio.run(template_manager.retrieve_templates(persona, stage))
                            if templates:
                                # Convert lead data to expected format
                                lead_data = {
                                    'fullName': name,
                                    'firstName': first_name,
                                    'jobTitle': title,
                                    'companyName': company,
                                    'email': email
                                }
                                result = asyncio.run(template_manager.generate_email(lead_data, templates))
                                if result["status"] == "success":
                                    st.session_state[f"subject_{lead_idx}"] = result["subject"]
                                    st.session_state[f"body_{lead_idx}"] = result["body"]
                                    st.success("✅ Email generated from templates!")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to generate email: {result['message']}")
                            else:
                                st.warning("No templates found for this persona and stage")
                        except Exception as e:
                            st.error(f"Error using template: {str(e)}")
            
            with btn_col2:
                if st.button("💾 Save Email", key=f"save_{lead_idx}"):
                    with st.spinner("Saving email..."):
                        try:
                            template_manager = SimpleEmailManager()
                            draft_result = asyncio.run(template_manager.save_draft(
                                lead_id=lead.get('id', str(lead_idx)),
                                subject=st.session_state.get(f"subject_{lead_idx}", ""),
                                body=st.session_state.get(f"body_{lead_idx}", ""),
                                persona=persona,
                                stage=stage
                            ))
                            
                            if draft_result["status"] == "success":
                                # Mark as template
                                template_result = asyncio.run(template_manager.mark_as_template(
                                    draft_id=draft_result["data"]["id"],
                                    persona=persona,
                                    stage=stage
                                ))
                                
                                if template_result["status"] == "success":
                                    st.success("✅ Saved as template for future use!")
                                else:
                                    st.error(f"Failed to mark as template: {template_result['message']}")
                            else:
                                st.error(f"Failed to save email: {draft_result['message']}")
                        except Exception as e:
                            st.error(f"Error saving email: {str(e)}")
            
            # Email composition fields
            subject = st.text_input(
                "Subject", 
                value=st.session_state.get(f"subject_{lead_idx}", f"Quick question about {company}"), 
                key=f"subject_{lead_idx}"
            )
            
            # Default email template
            default_template = f"""Hi {first_name},

I noticed your role as {title} at {company}. 

[Your personalized message here]

Best regards,
[Your name]"""
            
            email_body = st.text_area(
                "Body", 
                value=st.session_state.get(f"body_{lead_idx}", default_template),
                height=200, 
                key=f"body_{lead_idx}"
            )
            
            # Send email button
            if st.button("📤 Send Email", key=f"send_{lead_idx}", type="primary"):
                with st.spinner("Sending email..."):
                    result = email_manager.send_email(email, subject, email_body)
                    
                    if result["status"] == "success":
                        st.success("✅ Email sent successfully!")
                        # Mark as sent
                        st.session_state[sent_key] = True
                        st.session_state[compose_key] = False
                        st.session_state[email_key] = False
                        st.rerun()
                    else:
                        st.error(f"Failed to send email: {result['message']}")
        
        # Close button
        if st.button("❌ Close", key=f"close_modal_{lead_idx}"):
            st.session_state[email_key] = False
            st.rerun()
    
    st.title("📊 Leads Database")
    st.markdown("View and manage all scraped leads from the database")
    
    # Initialize Supabase client
    try:
        supabase_client = get_supabase_client()
        if not supabase_client:
            st.error("❌ Supabase client not initialized. Cannot display leads database.")
            return
    except Exception as e:
        st.error(f"❌ Error connecting to Supabase: {str(e)}")
        return
    
    # Initialize email manager
    email_manager = EmailManager(RESEND_API_KEY, "resend.dev", SENDER_EMAIL)
    
    try:
        # Get total leads count
        leads_result = supabase_client.table("leads") \
            .select("*") \
            .order("created_at", desc=True) \
            .execute()
            
        if not leads_result.data:
            st.info("No leads found in the database.")
            return
            
        total_leads = len(leads_result.data)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Leads", total_leads)
        with col2:
            # Count leads with emails
            leads_with_email = sum(1 for lead in leads_result.data if lead.get('email') or lead.get('email_address'))
            st.metric("Leads with Email", leads_with_email)
        with col3:
            # Average ICP score
            icp_scores = [float(lead.get('icp_percentage', 0)) for lead in leads_result.data if lead.get('icp_percentage')]
            avg_icp = sum(icp_scores) / len(icp_scores) if icp_scores else 0
            st.metric("Avg ICP Score", f"{avg_icp:.1f}%")
        
        st.markdown("---")
        
        # Convert to DataFrame for easier manipulation
        leads_df = pd.DataFrame(leads_result.data)
        
        # Add filters and sorting
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            # ICP Score filter
            min_icp = st.slider(
                "Min ICP Score",
                min_value=0,
                max_value=100,
                value=0,
                key="icp_filter"
            )
        
        with col2:
            # Company filter
            companies = ["All"] + sorted(leads_df['company_name'].dropna().unique().tolist())
            company_filter = st.selectbox(
                "Company",
                options=companies,
                key="company_filter"
            )
        
        with col3:
            # Job title filter
            job_titles = ["All"] + sorted(leads_df['job_title'].dropna().unique().tolist())
            job_filter = st.selectbox(
                "Job Title",
                options=job_titles,
                key="job_filter"
            )
        
        with col4:
            # Email availability filter
            email_availability_filter = st.selectbox(
                "Email Availability",
                options=["All", "With Email", "Without Email"],
                index=1,  # Default to "With Email"
                key="email_availability_filter"
            )
        
        with col5:
            # Email status filter
            email_status_filter = st.selectbox(
                "Email Status",
                options=["All", "Sent", "Not Sent"],
                index=0,
                key="email_status_filter"
            )
        
        with col6:
            # Sort by filter
            sort_options = {
                "Newest First": ("created_at", False),
                "Oldest First": ("created_at", True),
                "Name A-Z": ("full_name", True),
                "Name Z-A": ("full_name", False),
                "Company A-Z": ("company_name", True),
                "Company Z-A": ("company_name", False),
                "ICP Score High-Low": ("icp_percentage", False),
                "ICP Score Low-High": ("icp_percentage", True)
            }
            sort_filter = st.selectbox(
                "Sort By",
                options=list(sort_options.keys()),
                index=0,
                key="sort_filter"
            )
        
        # Apply filters
        filtered_df = leads_df.copy()
        
        # Filter by ICP score
        if 'icp_percentage' in filtered_df.columns:
            filtered_df['icp_percentage'] = pd.to_numeric(filtered_df['icp_percentage'], errors='coerce')
            filtered_df = filtered_df[filtered_df['icp_percentage'].notna() & (filtered_df['icp_percentage'] >= min_icp)]
        
        # Filter by company
        if company_filter != "All":
            filtered_df = filtered_df[filtered_df['company_name'] == company_filter]
        
        # Filter by job title
        if job_filter != "All":
            filtered_df = filtered_df[filtered_df['job_title'] == job_filter]
        
        # Helper functions for email validation
        def has_valid_email(lead):
            try:
                email = lead.get('email') or lead.get('email_address', '')
                if pd.isna(email) or email is None:
                    return False
                email_str = str(email).strip()
                return email_str and email_str != 'No email' and email_str != 'nan' and '@' in email_str
            except:
                return False
        
        def has_no_valid_email(lead):
            return not has_valid_email(lead)
        
        # Filter by email availability
        if email_availability_filter == "With Email":
            email_mask = filtered_df.apply(has_valid_email, axis=1)
            # Handle any NaN values in the mask
            email_mask = email_mask.fillna(False)
            filtered_df = filtered_df[email_mask]
        elif email_availability_filter == "Without Email":
            email_mask = filtered_df.apply(has_no_valid_email, axis=1)
            # Handle any NaN values in the mask
            email_mask = email_mask.fillna(True)
            filtered_df = filtered_df[email_mask]
        
        # Filter by email status
        if email_status_filter != "All":
            if email_status_filter == "Sent":
                # Filter for leads that have been sent emails (check session state)
                sent_leads = []
                for _, lead in filtered_df.iterrows():
                    sent_key = f"email_sent_{lead.get('id', '')}"
                    if st.session_state.get(sent_key, False):
                        sent_leads.append(lead.name)
                if sent_leads:
                    filtered_df = filtered_df.loc[sent_leads]
                else:
                    filtered_df = filtered_df.iloc[0:0]  # Empty dataframe
            elif email_status_filter == "Not Sent":
                # Filter for leads that have NOT been sent emails
                not_sent_leads = []
                for _, lead in filtered_df.iterrows():
                    sent_key = f"email_sent_{lead.get('id', '')}"
                    if not st.session_state.get(sent_key, False):
                        not_sent_leads.append(lead.name)
                if not_sent_leads:
                    filtered_df = filtered_df.loc[not_sent_leads]
                else:
                    filtered_df = filtered_df.iloc[0:0]  # Empty dataframe
        
        # Display filtered results count
        st.info(f"📊 Showing {len(filtered_df)} of {len(leads_df)} leads")
        
        if filtered_df.empty:
            st.warning("No leads match your current filters.")
        else:
            # Display leads in table format
            st.subheader("📋 Leads Table")
            
            # Create table headers
            header_cols = st.columns([2.5, 2, 2, 1.5, 1.5, 1, 1.5])
            with header_cols[0]:
                st.markdown("**👤 Name & Title**")
            with header_cols[1]:
                st.markdown("**🏢 Company**")
            with header_cols[2]:
                st.markdown("**📧 Email**")
            with header_cols[3]:
                st.markdown("**📍 Location**")
            with header_cols[4]:
                st.markdown("**📱 Social Media**")
            with header_cols[5]:
                st.markdown("**🎯 ICP**")
            with header_cols[6]:
                st.markdown("**⚡ Actions**")
            
            st.markdown("---")
            
            # Display each lead as a table row
            for lead_idx, lead in filtered_df.iterrows():
                # Handle different field naming conventions
                name = lead.get('full_name') or lead.get('fullName', 'Unknown')
                first_name = lead.get('first_name') or lead.get('firstName', name.split()[0] if name != 'Unknown' else '')
                title = lead.get('job_title') or lead.get('jobTitle', 'Unknown Title')
                company = lead.get('company_name') or lead.get('companyName', 'Unknown Company')
                industry = lead.get('company_industry') or lead.get('companyIndustry', 'Unknown Industry')
                linkedin_url = lead.get('linkedin_url', '')
                email = lead.get('email') or lead.get('email_address', 'No email')
                
                # Location
                location_parts = []
                if lead.get('city'):
                    location_parts.append(str(lead.get('city')))
                if lead.get('state'):
                    location_parts.append(str(lead.get('state')))
                if lead.get('country'):
                    location_parts.append(str(lead.get('country')))
                location = ', '.join(location_parts) if location_parts else 'N/A'
                
                # ICP Score
                icp_score = lead.get('icp_percentage') or lead.get('icp_score', 0)
                icp_grade = lead.get('icp_grade', 'N/A')
                
                # Create table row
                row_cols = st.columns([2.5, 2, 2, 1.5, 1.5, 1, 1.5])
                
                with row_cols[0]:
                     # Create name with LinkedIn link
                     if linkedin_url:
                         st.markdown(f"**[{name}]({linkedin_url})**")
                     else:
                         st.markdown(f"**{name}**")
                     st.caption(title)
                
                with row_cols[1]:
                    # Create company with LinkedIn and website links
                    company_links = []
                    
                    # Add company LinkedIn URL if available
                    company_linkedin = lead.get('company_linkedin') or lead.get('companyLinkedin')
                    if company_linkedin:
                        if not company_linkedin.startswith(('http://', 'https://')):
                            company_linkedin = f"https://{company_linkedin}"
                        company_links.append(f"[{company}]({company_linkedin})")
                    else:
                        company_links.append(company)
                    
                    # Add company website link if available
                    company_website = lead.get('company_website') or lead.get('companyWebsite')
                    if company_website:
                        if not company_website.startswith(('http://', 'https://')):
                            company_website = f"https://{company_website}"
                        company_links.append(f"[🌐]({company_website})")
                    
                    st.markdown(f"**{' '.join(company_links)}**")
                    st.caption(industry)
                
                with row_cols[2]:
                    if email and email != 'No email':
                        st.markdown(f"📧 {email}")
                    else:
                        st.caption("No email")
                
                with row_cols[3]:
                    st.caption(location)
                
                with row_cols[4]:
                    # Social Media Links
                    social_links = []
                    
                    # Add Twitter link if available
                    company_twitter = lead.get('company_twitter') or lead.get('companyTwitter')
                    if company_twitter:
                        if not company_twitter.startswith(('http://', 'https://')):
                            company_twitter = f"https://twitter.com/{company_twitter.lstrip('@')}"
                        social_links.append(f"[🐦]({company_twitter})")
                    
                    # Add Facebook link if available
                    company_facebook = lead.get('company_facebook') or lead.get('companyFacebook')
                    if company_facebook:
                        if not company_facebook.startswith(('http://', 'https://')):
                            company_facebook = f"https://facebook.com/{company_facebook}"
                        social_links.append(f"[📘]({company_facebook})")
                    
                    if social_links:
                        st.markdown(' '.join(social_links))
                    else:
                        st.caption("No social media")
                
                with row_cols[5]:
                    if icp_score >= 80:
                        st.success(f"{icp_score}%")
                    elif icp_score >= 60:
                        st.warning(f"{icp_score}%")
                    else:
                        st.info(f"{icp_score}%")
                    st.caption(icp_grade)
                
                with row_cols[6]:
                    # Email management button
                    if email and email != 'No email':
                        email_key = f"email_modal_{lead.get('id', lead_idx)}"
                        if st.button("📧 Email", key=f"email_btn_{lead_idx}", help="Manage email for this lead"):
                            st.session_state[email_key] = True
                            st.session_state[f"current_lead_name_{lead_idx}"] = name
                            st.session_state[f"current_lead_title_{lead_idx}"] = title
                            st.session_state[f"current_lead_company_{lead_idx}"] = company
                            st.session_state[f"current_lead_email_{lead_idx}"] = email
                            st.session_state[f"current_lead_first_name_{lead_idx}"] = first_name
                            st.session_state[f"current_lead_data_{lead_idx}"] = lead
                        
                        # Email management modal
                        if st.session_state.get(email_key, False):
                            # Get stored lead data
                            stored_name = st.session_state.get(f"current_lead_name_{lead_idx}", name)
                            stored_title = st.session_state.get(f"current_lead_title_{lead_idx}", title)
                            stored_company = st.session_state.get(f"current_lead_company_{lead_idx}", company)
                            stored_email = st.session_state.get(f"current_lead_email_{lead_idx}", email)
                            stored_first_name = st.session_state.get(f"current_lead_first_name_{lead_idx}", first_name)
                            stored_lead = st.session_state.get(f"current_lead_data_{lead_idx}", lead)
                            
                            email_modal(stored_name, stored_title, stored_company, stored_email, stored_first_name, stored_lead, lead_idx, email_key)
                    else:
                        st.caption("No email")
                
                # Add separator between rows
                if lead_idx < len(filtered_df) - 1:
                    st.markdown("---")
                    
    except Exception as e:
        st.error(f"⚠️ Error fetching data from database: {str(e)}")


if __name__ == "__main__":
    main()
