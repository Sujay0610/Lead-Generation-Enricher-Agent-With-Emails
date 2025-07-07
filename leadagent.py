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
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
import hashlib

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
            model="openai/gpt-4.1-nano-2025-04-14",
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
                    "to": "sujay0620@gmail.com",
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
    """Enhanced tool for scraping leads with ICP scoring using Apollo.io"""

    def __init__(self, apify_token_or_tokens, sheets_service):
        self.sheets_service = sheets_service
        if isinstance(apify_token_or_tokens, list):
            self.token_queue = APIKeyQueue(apify_token_or_tokens)
        else:
            self.token_queue = APIKeyQueue([apify_token_or_tokens])
        self.ai_icp_scorer = AIICPScorer(OPENAI_API_KEY)
        
        # Initialize or load used keys from session state
        if "used_apify_keys" not in st.session_state:
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
        self.exhausted_keys = st.session_state.exhausted_apify_keys

    def get_next_unused_key(self):
        """Get the next unused API key. If all keys have been used, reset and start over."""
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

    def search_apollo_profiles(self, query: str, num_results: int = 50) -> List[Dict]:
        """Search for profiles using Apify Apollo scraper"""
        try:
            st.info(f"🔍 Starting Apollo.io scraping with query: {query}")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("⏳ Waiting for Apollo.io scraper to initialize (this may take a few minutes)...")

            # Use the next available unused API key
            url = "https://api.apify.com/v2/acts/iJcISG5H8FJUSRoVA/run-sync-get-dataset-items"
            payload = {
                "contact_email_exclude_catch_all": True,
                "contact_email_status_v2": True,
                "max_result": num_results,
                "url": query
            }

            # Try up to 3 different API keys if needed
            max_retries = min(3, self.total_keys)
            response = None
            
            for attempt in range(max_retries):
                try:
                    api_key = self.get_next_unused_key()
                    st.info(f"🔑 Using API key: {api_key[:10]}... ({len(self.used_keys)}/{self.total_keys} keys used today, {len(self.exhausted_keys)} exhausted)")
                    
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }

                    # Make the request with a longer timeout
                    response = requests.post(url, json=payload, headers=headers, timeout=600)  # 10 minute timeout
                    
                    # Check for daily limit exhausted error
                    try:
                        resp_json = response.json()
                        
                        # Check for the specific error message about daily limit
                        if isinstance(resp_json, list) and len(resp_json) == 1 and "message" in resp_json[0]:
                            error_msg = resp_json[0]["message"]
                            if "exhausted their daily run limit" in error_msg:
                                # Mark this key as exhausted
                                self.mark_key_exhausted(api_key)
                                
                                if attempt < max_retries - 1:
                                    st.warning(f"API key {api_key[:10]}... has reached its daily limit. Trying another key...")
                                    continue
                                else:
                                    st.error("All available API keys have reached their daily limit.")
                                    return []
                    except:
                        # If we can't parse the JSON or check for the error, continue with normal flow
                        pass
                    
                    # Handle both 200 and 400 responses since the scraper might return data even with 400
                    if response.status_code not in [200, 201, 400]:
                        if attempt < max_retries - 1:
                            st.warning(f"API key {api_key[:10]}... failed with status code {response.status_code}. Trying another key...")
                            continue
                        else:
                            st.error(f"Apollo.io API returned unexpected status code {response.status_code} after {max_retries} attempts")
                            return []
                    
                    # If we got here, the request was successful
                    break
                    
                except requests.RequestException as e:
                    if attempt < max_retries - 1:
                        st.warning(f"API request failed: {str(e)}. Trying another key...")
                        continue
                    else:
                        st.error(f"API request failed after {max_retries} attempts: {str(e)}")
                        return []
            
            # If all retries failed and we don't have a response
            if response is None:
                st.error("Failed to get a response from Apollo.io after multiple attempts")
                return []
                
            # Process the response
            try:
                results = response.json()

                # Check if results is empty or not a list
                if not results:
                    st.warning("No results returned from Apollo.io. The search might be too narrow or the credits might be exhausted.")
                    return []
                
                # Check for the specific error message about daily limit again
                if isinstance(results, list) and len(results) == 1 and "message" in results[0]:
                    error_msg = results[0]["message"]
                    if "exhausted their daily run limit" in error_msg:
                        # Mark the current key as exhausted
                        self.mark_key_exhausted(api_key)
                        st.error(f"All available API keys have reached their daily limit: {error_msg}")
                        return []
                
                # If results is a dict with an error message, check for data
                if isinstance(results, dict):
                    if 'data' in results:
                        results = results['data']
                    elif 'items' in results:
                        results = results['items']
                    else:
                        st.warning("Unexpected response format from Apollo.io")
                        return []
                
                # Ensure results is a list
                if not isinstance(results, list):
                    st.warning("Invalid response format from Apollo.io")
                    return []

            except json.JSONDecodeError:
                st.error("Invalid JSON response from Apollo.io API")
                return []
            
            st.success(f"✅ Found {len(results)} profiles from Apollo.io")

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
                
            return profiles[:num_results]

        except Exception as e:
            st.error(f"Error in Apollo.io scraping: {str(e)}")
            return []

    def scrape_leads(self, query_json: str) -> str:
        """Full pipeline: search → save → score → update."""
        try:
            st.info("🚀 Starting lead generation process...")

            # Step 1: Parse input and validate
            try:
                input_data = json.loads(query_json)
                if isinstance(input_data, dict) and "query" in input_data and isinstance(input_data["query"], list):
                    params_data = input_data["query"][0]
                else:
                    params_data = input_data

                # Add default employee ranges if not specified
                if isinstance(params_data, dict):
                    if not params_data.get("employee_ranges"):
                        params_data["employee_ranges"] = ["11,20", "21,50", "51,100", "101,200"]
                    if not params_data.get("sort_field"):
                        params_data["sort_field"] = "recommendations_score"
                    if not params_data.get("sort_ascending"):
                        params_data["sort_ascending"] = "false"

                if not isinstance(params_data, list):
                    params_data = [params_data]
            except json.JSONDecodeError as e:
                return f"Invalid query JSON: {str(e)}"

            # Step 2: Scrape leads from Apollo
            st.subheader("🔍 Step 1: Scraping Leads from Apollo")
            all_profiles = []
            for q in params_data:
                apollo_url = self.generate_apollo_url(q)
                st.info(f"Generated Apollo.io search URL: {apollo_url}")
                
                profiles = self.search_apollo_profiles(apollo_url, num_results=st.session_state["leads_per_query"])
                if profiles:
                    all_profiles.extend(profiles)

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

            # Step 4: Update sheets with scores
            st.subheader("📊 Step 3: Updating with ICP Scores")
            if scored_profiles:
                try:
                    final_save_msg = self.save_enriched_data_to_sheets(scored_profiles)
                    st.success("✅ ICP scores saved to Google Sheets")
                except Exception as e:
                    st.error(f"Error saving ICP scores: {str(e)}")
                    return "Failed to save ICP scores to Google Sheets"

            # Final summary
            successful_scores = [p for p in scored_profiles if p.get("icp_score") is not None]
            summary = (
                "🎉 **Lead Generation Complete!**\n\n"
                f"📊 **Results Summary:**\n"
                f"- Apollo.io profiles found: {total_leads}\n"
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
                # Verify headers match
                existing_headers = current_values[0]
                if existing_headers != expected_columns:
                    # Clear the sheet and add correct headers
                    sheet.clear()
                    sheet.append_row(expected_columns)
                    existing_keys = set()
                else:
                    # Get existing linkedin_urls for deduplication
                    if len(current_values) > 1:
                        existing_rows = pd.DataFrame(current_values[1:], columns=existing_headers)
                        existing_keys = set(existing_rows["linkedin_url"].str.strip().str.lower())
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
            successful_scores = df_to_upload[df_to_upload['scraping_status'] == 'success']
            if not successful_scores.empty:
                avg_score = pd.to_numeric(successful_scores['icp_percentage'], errors='coerce').mean()
                grade_counts = successful_scores['icp_grade'].value_counts().to_dict()
                
                summary = f"Successfully appended {len(df_to_upload)} new enriched profiles!\n"
                summary += f"Average ICP Score: {avg_score:.1f}%\n"
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
    col1, col2 = st.columns(2)
    
    with col1:
        min_score = st.slider("Minimum ICP Score (%)", 0, 100, 0)
    
    with col2:
        email_status_filter = st.selectbox(
            "Email Status",
            options=["All", "Sent", "Not Sent"],
            index=0
        )
    
    # Sorting options
    sort_by = st.selectbox(
        "Sort by",
        options=["ICP Score (Best to Worst)", "ICP Score (Worst to Best)"],
        index=0
    )

    # Bulk email section
    st.markdown("---")
    st.subheader("📨 Bulk Email Options")
    
    # Initialize email manager
    email_manager = EmailManager(RESEND_API_KEY, "resend.dev", SENDER_EMAIL)
    
    # Calculate stats for unsent emails
    unsent_leads = []
    for idx, lead in leads_df.iterrows():
        email_key = f"email_sent_{hashlib.md5(lead.get('linkedin_url', '').encode()).hexdigest()}"
        if not st.session_state.get(email_key, False):
            unsent_leads.append(lead)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"📊 {len(unsent_leads)} leads haven't been emailed yet")
        
    with col2:
        if len(unsent_leads) > 0:
            if st.button("🚀 Generate & Send All", help="Generate and send emails to all leads that haven't been contacted"):
                bulk_progress = st.progress(0)
                status_text = st.empty()
                
                successful_sends = 0
                failed_sends = 0
                
                for idx, lead in enumerate(unsent_leads):
                    try:
                        status_text.text(f"Processing {idx + 1}/{len(unsent_leads)}: {lead.get('fullName', 'Unknown')}")
                        
                        # Prepare lead data
                        lead_data = {
                            "fullName": lead.get('fullName', ''),
                            "firstName": lead.get('firstName', ''),
                            "lastName": lead.get('lastName', ''),
                            "email": lead.get('email', ''),
                            "jobTitle": lead.get('jobTitle', ''),
                            "companyName": lead.get('companyName', ''),
                            "companyIndustry": lead.get('companyIndustry', ''),
                            "companyWebsite": lead.get('companyWebsite', ''),
                            "companyLinkedin": lead.get('companyLinkedin', ''),
                            "linkedin_url": lead.get('linkedin_url', ''),
                            "location": lead.get('location', ''),
                            "headline": lead.get('headline', ''),
                            "about": lead.get('about', ''),
                            "icp_score": lead.get('icp_percentage', 0),
                            "icp_grade": lead.get('icp_grade', 'N/A')
                        }
                        
                        # Generate email
                        result = email_manager.generate_and_preview_email(lead_data)
                        
                        if result["status"] == "success":
                            # Send email
                            send_result = email_manager.send_email(
                                result["to_email"],
                                result["subject"],
                                result["body"]
                            )
                            
                            if send_result["status"] == "success":
                                # Mark as sent in session state
                                email_key = f"email_sent_{hashlib.md5(lead.get('linkedin_url', '').encode()).hexdigest()}"
                                st.session_state[email_key] = True
                                
                                # Update the email_status in Google Sheets
                                try:
                                    # Get the sheet
                                    ss = sheets_service.open("Leadgen")
                                    sheet = ss.worksheet("Sheet1")
                                    
                                    # Find the row with this LinkedIn URL
                                    linkedin_url = lead.get('linkedin_url', '')
                                    if linkedin_url:
                                        # Get all LinkedIn URLs
                                        linkedin_urls = sheet.col_values(1)  # Assuming LinkedIn URL is in column A
                                        if linkedin_url in linkedin_urls:
                                            row_idx = linkedin_urls.index(linkedin_url) + 1  # +1 because sheets are 1-indexed
                                            # Find the email_status column
                                            headers = sheet.row_values(1)
                                            if 'email_status' in headers:
                                                col_idx = headers.index('email_status') + 1  # +1 because sheets are 1-indexed
                                                # Update the cell
                                                sheet.update_cell(row_idx, col_idx, "Sent")
                                except Exception as sheet_e:
                                    st.warning(f"Could not update email status in sheet: {str(sheet_e)}")
                                
                                successful_sends += 1
                            else:
                                failed_sends += 1
                        else:
                            failed_sends += 1
                        
                        # Update progress
                        bulk_progress.progress((idx + 1) / len(unsent_leads))
                        
                    except Exception as e:
                        st.error(f"Error processing {lead.get('fullName', 'Unknown')}: {str(e)}")
                        failed_sends += 1
                    
                    # Small delay to avoid rate limits
                    time.sleep(1)
                
                # Show final results
                if successful_sends > 0:
                    st.success(f"✅ Successfully sent {successful_sends} emails!")
                if failed_sends > 0:
                    st.error(f"❌ Failed to send {failed_sends} emails")
                    
                # Clear progress
                bulk_progress.empty()
                status_text.empty()
                
                # Refresh the page to update UI
                st.rerun()
    
    # Display leads in a compact format
    st.markdown("---")
    st.subheader(f"📋 Individual Lead Management")
    
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
    
    if filtered_df.empty:
        st.info("No leads match your filters.")
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
                st.markdown(f"🏭 {industry} • 📍 {lead.get('location', 'N/A')}")
            
            with col2:
                icp_score = lead.get('icp_percentage', 0)
                icp_grade = lead.get('icp_grade', 'N/A')
                st.markdown(f"**ICP:** {icp_score}% ({icp_grade})")
                st.markdown(f"📧 {lead.get('email', 'No email')}")
            
            with col3:
                # Email status tracking
                email_key = f"email_sent_{hashlib.md5(lead.get('linkedin_url', '').encode()).hexdigest()}"
                preview_key = f"preview_{email_key}"
                
                if st.session_state.get(email_key, False):
                    st.success("✅ Sent")
                    if st.button("📧 Send Again", key=f"resend_{idx}", help="Send another email"):
                        st.session_state[email_key] = False
                        st.session_state[preview_key] = False
                        st.rerun()
                else:
                    if not st.session_state.get(preview_key, False):
                        if st.button("📧 Generate", key=f"generate_{idx}", help="Generate email"):
                            # Prepare lead data
                            lead_data = {
                                "fullName": lead.get('fullName', ''),
                                "firstName": lead.get('firstName', ''),
                                "lastName": lead.get('lastName', ''),
                                "email": lead.get('email', ''),
                                "jobTitle": lead.get('jobTitle', ''),
                                "companyName": lead.get('companyName', ''),
                                "companyIndustry": lead.get('companyIndustry', ''),
                                "companyWebsite": lead.get('companyWebsite', ''),
                                "companyLinkedin": lead.get('companyLinkedin', ''),
                                "companySize": lead.get('companySize', ''),
                                "linkedin_url": lead.get('linkedin_url', ''),
                                "location": lead.get('location', ''),
                                "headline": lead.get('headline', ''),
                                "about": lead.get('about', ''),
                                "icp_score": lead.get('icp_percentage', 0),
                                "icp_grade": lead.get('icp_grade', 'N/A')
                            }
                            
                            # Generate email preview
                            with st.spinner("Generating email..."):
                                result = email_manager.generate_and_preview_email(lead_data)
                                
                                if result["status"] == "success":
                                    st.session_state[f"email_preview_{idx}"] = result
                                    st.session_state[preview_key] = True
                                    st.rerun()
                                else:
                                    st.error(f"Failed to generate email: {result['message']}")
                    else:
                        # Show email preview in a more compact format
                        preview = st.session_state[f"email_preview_{idx}"]
                        st.markdown("#### 📨 Email Preview")
                        
                        # Create two columns with custom widths for the email preview
                        email_col1, email_col2 = st.columns([4, 1])
                        
                        with email_col1:
                            st.text_input("Subject", value=preview["subject"], key=f"subject_{idx}")
                            email_body = st.text_area("Body", value=preview["body"], height=300, key=f"body_{idx}")
                        
                        with email_col2:
                            st.markdown("<br><br>", unsafe_allow_html=True)  # Add some spacing
                            if st.button("Regenerate", key=f"regenerate_{idx}", help="Regenerate email"):
                                st.session_state[preview_key] = False
                                st.rerun()
                            
                            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing between buttons
                            if st.button("✉️ Send", key=f"send_{idx}"):
                                with st.spinner("Sending..."):
                                    result = email_manager.send_email(
                                        preview["to_email"],
                                        st.session_state[f"subject_{idx}"],
                                        st.session_state[f"body_{idx}"]
                                    )
                                    
                                    if result["status"] == "success":
                                        st.success("Sent!")
                                        st.session_state[email_key] = True
                                        st.session_state[preview_key] = False
                                        
                                        # Update the email_status in Google Sheets
                                        try:
                                            # Get the sheet
                                            ss = sheets_service.open("Leadgen")
                                            sheet = ss.worksheet("Sheet1")
                                            
                                            # Find the row with this LinkedIn URL
                                            linkedin_url = lead.get('linkedin_url', '')
                                            if linkedin_url:
                                                # Get all LinkedIn URLs
                                                linkedin_urls = sheet.col_values(1)  # Assuming LinkedIn URL is in column A
                                                if linkedin_url in linkedin_urls:
                                                    row_idx = linkedin_urls.index(linkedin_url) + 1  # +1 because sheets are 1-indexed
                                                    # Find the email_status column
                                                    headers = sheet.row_values(1)
                                                    if 'email_status' in headers:
                                                        col_idx = headers.index('email_status') + 1  # +1 because sheets are 1-indexed
                                                        # Update the cell
                                                        sheet.update_cell(row_idx, col_idx, "Sent")
                                        except Exception as sheet_e:
                                            st.warning(f"Could not update email status in sheet: {str(sheet_e)}")
                                        
                                        st.rerun()
                                    else:
                                        st.error(f"Failed: {result['message']}")

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


def create_lead_agent():
    """Create the lead generation agent"""
    
    sheets_service = setup_google_sheets()
    if not sheets_service:
        return None
    
    scraping_tool = LeadScrapingTool(APIFY_API_TOKENS, sheets_service)
    
    tools = [
        Tool(
            name="leadScraping",
            description="Use this tool to scrape leads into a Google Sheet. Only call this tool once you have enough information to complete the desired JSON search query.",
            func=scraping_tool.scrape_leads
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
        ["🎯 Lead Generation", "📧 Email Management", "⚙️ ICP Configuration"]
    )
    
    if page == "🎯 Lead Generation":
        lead_generation_page()
    elif page == "📧 Email Management":
        email_management_page()
    elif page == "⚙️ ICP Configuration":
        icp_configuration_page()

def lead_generation_page():
    """Lead generation page with chat interface and direct query option"""
    st.title("🎯 Lead Generation System")
    st.markdown("Powered by AI Agent + Web Scraping + Data Enrichment + ICP Scoring")
    
    # Add a toggle for input method
    input_method = st.radio(
        "Choose input method:",
        ["Chat with Lead Generation Joe", "Direct Query Form"],
        horizontal=True
    )
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Leads per query selector
        st.subheader("🎯 Query Settings")
        leads_per_query = st.selectbox(
            "Leads per search query",
            options=[20, 25, 50, 100],
            index=3,  # Default to 20
            help="Number of LinkedIn profiles to fetch per search query. More leads = longer processing time."
        )
        st.session_state["leads_per_query"] = leads_per_query
        
        # API Key management
        st.markdown("---")
        st.subheader("🔑 API Key Management")
        
        # Display current API key usage
        if "used_apify_keys" in st.session_state:
            used_count = len(st.session_state.used_apify_keys)
            total_count = len(APIFY_API_TOKENS)
            st.info(f"API Key Usage: {used_count}/{total_count} keys used today")
            
            # Add reset button for used keys
            if used_count > 0 and st.button("Reset Used API Keys"):
                st.session_state.used_apify_keys = set()
                st.success("✅ API key usage reset successfully!")
                st.experimental_rerun()
        
        # Display exhausted API keys
        if "exhausted_apify_keys" in st.session_state and len(st.session_state.exhausted_apify_keys) > 0:
            exhausted_count = len(st.session_state.exhausted_apify_keys)
            total_count = len(APIFY_API_TOKENS)
            st.warning(f"⚠️ {exhausted_count}/{total_count} API keys have reached daily limit")
            
            # Add reset button for exhausted keys
            if st.button("Reset Exhausted API Keys"):
                st.session_state.exhausted_apify_keys = set()
                st.success("✅ Exhausted API keys reset successfully!")
                st.experimental_rerun()
        
        # API Key checks
        st.markdown("---")
        st.subheader("🔑 API Status")
        api_status = {
            "OpenAI API": bool(OPENAI_API_KEY),
            "Apollo.io API": bool(APIFY_API_TOKENS),
            "Google Sheets": bool(GOOGLE_SHEETS_CREDENTIALS),
            "Resend API": bool(RESEND_API_KEY)
        }
        
        for service, status in api_status.items():
            if status:
                st.success(f"✅ {service}")
            else:
                st.error(f"❌ {service} - Please add to secrets")
        
        st.markdown("---")
        st.markdown("### How to use:")
        st.markdown("1. Start by saying 'Hi' to Lead Generation Joe")
        st.markdown("2. Provide locations, businesses, and job titles")
        st.markdown("3. Joe will automatically:")
        st.markdown("   - 🔍 Search Google for LinkedIn profiles")
        st.markdown("   - 🤖 Enrich profiles with Apify scraper")
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
        
        st.subheader("🔍 Direct Lead Search")
        st.markdown("""
        Specify your search criteria to generate leads directly from Apollo.io:
        
        1. Select job titles, locations, and industries from the dropdown menus
        2. Add custom values if needed
        3. Choose company size ranges
        4. Click 'Generate Leads' to start the search
        
        Each search will retrieve up to 25 leads. For best results, use specific locations and job titles.
        """)
        
        with st.form("direct_lead_search_form"):
            # Job Title(s) - Multi-select with common options and custom input
            st.subheader("👔 Job Titles")
            preset_titles = st.multiselect(
                "Select job titles:",
                [
                    "Operations Head", "Operations Manager", "Plant Manager", "Production Engineer",
                    "Facility Manager", "Service Head", "Asset Manager",
                    "Maintenance Manager", "Operations Director", "COO"
                ]
            )
            
            custom_title = st.text_input("Add custom job title:", "")
            
            # Location(s) - Multi-select with common options and custom input
            st.subheader("📍 Locations")
            preset_locations = st.multiselect(
                "Select locations:",
                [
                    "United States", "Canada", "United Kingdom", "Australia", "Singapore", "India"
                ]
            )
            
            custom_location = st.text_input("Add custom location:", "")
            
            # Industries - Multi-select with common options and custom input
            st.subheader("🏭 Industries")
            preset_industries = st.multiselect(
                "Select industries:",
                [
                    "Manufacturing", "Industrial Automation", "Consumer Electronics"
                ]
            )
            
            custom_industry = st.text_input("Add custom industry:", "")
            
            # Company Size - Multi-select of ranges
            st.subheader("🏢 Company Size")
            company_sizes = st.multiselect(
                "Select company size ranges:",
                [
                    "1,10", "11,20", "21,50", "51,100", "101,200"
                ],
                default=["1,10", "11,20", "21,50", "51,100"]
            )
            
            # Submit button
            submit_button = st.form_submit_button("🚀 Generate Leads")
        
        if submit_button:
            # Create query from selections
            job_titles = preset_titles.copy()
            if custom_title:
                job_titles.append(custom_title)
            
            locations = preset_locations.copy()
            if custom_location:
                locations.append(custom_location)
            
            industries = preset_industries.copy()
            if custom_industry:
                industries.append(custom_industry)
            
            # Check if we have enough info to proceed
            if not job_titles or not locations or not industries:
                st.error("Please provide at least one job title, one location, and one industry.")
                return
            
            # Format for the query
            formatted_job_titles = [title.replace(" ", "+") for title in job_titles]
            formatted_locations = [location.replace(" ", "+") for location in locations]
            formatted_industries = [industry.replace(" ", "+") for industry in industries]
            
            # Ensure company sizes are in the correct format (already in comma format)
            formatted_company_sizes = company_sizes
            
            # Create query JSON
            query_json = json.dumps({
                "query": [{
                    "job_title": formatted_job_titles,
                    "location": formatted_locations,
                    "business": formatted_industries,
                    "employee_ranges": formatted_company_sizes
                }]
            })
            
            # Create a scraping tool instance
            scraping_tool = LeadScrapingTool(APIFY_API_TOKENS, sheets_service)
            
            # Execute the scraping process
            with st.spinner("🔍 Generating leads from Apollo.io..."):
                try:
                    result = scraping_tool.scrape_leads(query_json)
                    st.success("✅ Lead generation completed!")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"❌ Error generating leads: {str(e)}")

if __name__ == "__main__":
    main()
