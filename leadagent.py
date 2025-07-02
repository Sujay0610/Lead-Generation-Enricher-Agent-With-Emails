import streamlit as st
import json
import re
from typing import List, Dict, Any
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
from langgraph.checkpoint.memory import MemorySaver  # Added missing import


# Configuration
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = st.secrets.get("GOOGLE_CSE_ID", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
APIFY_API_TOKEN = st.secrets.get("APIFY_API_TOKEN", "")
GOOGLE_SHEETS_CREDENTIALS = st.secrets.get("GOOGLE_SHEETS_CREDENTIALS", {})

# class LeadScrapingTool:
#     """Tool for scraping leads using Google Custom Search API and enriching with Apify"""
    
#     def __init__(self, api_key: str, cse_id: str, sheets_service, apify_token: str):
#         self.api_key = api_key
#         self.cse_id = cse_id
#         self.sheets_service = sheets_service
#         self.apify_token = apify_token
        
#     def search_linkedin_profiles(self, query: str, num_results: int = 3) -> List[Dict]:
#         """Search for LinkedIn profiles using Google Custom Search"""
#         results = []
#         start_index = 1
        
#         st.info(f"🔍 Searching for: {query}")
        
#         while len(results) < num_results:
#             try:
#                 url = "https://www.googleapis.com/customsearch/v1"
#                 params = {
#                     'key': self.api_key,
#                     'cx': self.cse_id,
#                     'q': query,
#                     'start': start_index,
#                     'num': min(10, num_results - len(results))
#                 }
                
#                 response = requests.get(url, params=params)
#                 response.raise_for_status()
#                 data = response.json()
                
#                 if 'items' not in data:
#                     break
                    
#                 for item in data['items']:
#                     if 'linkedin.com/in/' in item['link']:
#                         results.append({
#                             'title': item.get('title', ''),
#                             'link': item.get('link', ''),
#                             'snippet': item.get('snippet', ''),
#                             'found_at': datetime.now().isoformat()
#                         })
                
#                 start_index += 10
#                 time.sleep(0.1)  # Rate limiting
                
#             except Exception as e:
#                 st.error(f"Error searching: {str(e)}")
#                 break
        
#         st.success(f"✅ Found {len(results)} LinkedIn profiles")
#         return results
    
#     def enrich_profile_with_apify(self, linkedin_url: str) -> Dict:
#         """Enrich a single LinkedIn profile using Apify"""
#         try:
#             # Apify API call
#             url = "https://api.apify.com/v2/acts/dev_fusion~linkedin-profile-scraper/run-sync-get-dataset-items"
            
#             payload = {
#                 "profileUrls": [linkedin_url],
#                 "maxDelay": 5,
#                 "minDelay": 1
#             }
            
#             headers = {
#                 "Authorization": f"Bearer {self.apify_token}",
#                 "Content-Type": "application/json"
#             }
            
#             response = requests.post(url, json=payload, headers=headers, timeout=30)
#             response.raise_for_status()
            
#             data = response.json()
#             if data and len(data) > 0:
#                 profile = data[0]
#                 return {
#                     'linkedin_url': linkedin_url,
#                     'firstName': profile.get('firstName', ''),
#                     'lastName': profile.get('lastName', ''),
#                     'fullName': profile.get('fullName', ''),
#                     'headline': profile.get('headline', ''),
#                     'connections': profile.get('connectionsCount', ''),
#                     'followers': profile.get('followersCount', ''),
#                     'email': profile.get('email', ''),
#                     'mobileNumber': profile.get('mobileNumber', ''),
#                     'jobTitle': profile.get('jobTitle', ''),
#                     'companyName': profile.get('companyName', ''),
#                     'companyIndustry': profile.get('companyIndustry', ''),
#                     'companyWebsite': profile.get('companyWebsiteUrl', ''),
#                     'companyLinkedin': profile.get('companyLinkedinUrl', ''),
#                     'companyFoundedIn': profile.get('companyFoundedYear', ''),
#                     'companySize': profile.get('companySize', ''),
#                     'location': profile.get('location', ''),
#                     'about': profile.get('about', ''),
#                     'experience': json.dumps(profile.get('experience', [])),
#                     'education': json.dumps(profile.get('education', [])),
#                     'skills': json.dumps(profile.get('skills', [])),
#                     'industry': profile.get('industry', ''),
#                     'profile_picture': profile.get('profilePicture', ''),
#                     'scraped_at': datetime.now().isoformat(),
#                     'scraping_status': 'success'
#                 }
#             else:
#                 return {
#                     'linkedin_url': linkedin_url,
#                     'scraping_status': 'no_data',
#                     'scraped_at': datetime.now().isoformat()
#                 }
                
#         except Exception as e:
#             return {
#                 'linkedin_url': linkedin_url,
#                 'scraping_status': 'error',
#                 'error_message': str(e),
#                 'scraped_at': datetime.now().isoformat()
#             }
    
#     def batch_enrich_profiles(self, linkedin_urls: List[str]) -> List[Dict]:
#         """Enrich multiple LinkedIn profiles with progress tracking"""
#         enriched_profiles = []
#         total_profiles = len(linkedin_urls)
        
#         # Create progress bar
#         progress_bar = st.progress(0)
#         status_text = st.empty()
        
#         for i, url in enumerate(linkedin_urls):
#             status_text.text(f"🔍 Enriching profile {i+1}/{total_profiles}: {url}")
            
#             # Enrich profile
#             enriched_profile = self.enrich_profile_with_apify(url)
#             enriched_profiles.append(enriched_profile)
            
#             # Update progress
#             progress = (i + 1) / total_profiles
#             progress_bar.progress(progress)
            
#             # Rate limiting - be respectful to Apify
#             if i < total_profiles - 1:  # Don't sleep after last request
#                 time.sleep(2)  # 2 seconds between requests
        
#         status_text.text(f"✅ Completed enriching {total_profiles} profiles!")
#         return enriched_profiles
    
#     def save_enriched_data_to_sheets(self, enriched_data: List[Dict], sheet_name: str = "Sheet1") -> str:
#         """Append enriched lead data to Google Sheets without clearing existing data"""
#         try:
#             # Open existing spreadsheet and worksheet
#             spreadsheet = self.sheets_service.open("PyLeads")
#             sheet = spreadsheet.worksheet(sheet_name)

#             if enriched_data:
#                 df_new = pd.DataFrame(enriched_data).fillna('')

#                 # Get existing data
#                 existing_records = sheet.get_all_records()
#                 df_existing = pd.DataFrame(existing_records) if existing_records else pd.DataFrame()

#                 # Avoid duplicates based on linkedin_url
#                 if not df_existing.empty and 'linkedin_url' in df_existing.columns:
#                     df_new = df_new[~df_new['linkedin_url'].isin(df_existing['linkedin_url'])]

#                 if df_new.empty:
#                     return "No new profiles to append (all were duplicates)."

#                 # Calculate starting row to append
#                 start_row = len(existing_records) + 2  # +1 for header, +1 for 1-based indexing

#                 # Prepare data to append
#                 data_to_append = df_new.values.tolist()

#                 # Append row-by-row to avoid overwriting
#                 for i, row in enumerate(data_to_append):
#                     cell_range = f"A{start_row + i}"
#                     sheet.insert_row(row, index=start_row + i)

#                 return f"Successfully appended {len(df_new)} new enriched profiles to Google Sheets!"
#             else:
#                 return "No enriched data to save."

#         except Exception as e:
#             return f"Error appending enriched data to sheets: {str(e)}"

    
#     def scrape_leads(self, query_json: str) -> str:
#         """Main scraping and enriching function called by the agent"""
#         try:
#             st.info("🚀 Starting lead generation process...")
            
#             query_data = json.loads(query_json)
#             all_linkedin_urls = []
            
#             # Step 1: Search for LinkedIn profiles
#             st.subheader("📋 Step 1: Searching LinkedIn Profiles")
            
#             for query_obj in query_data:
#                 locations = query_obj.get('location', [])
#                 businesses = query_obj.get('business', [])
#                 job_titles = query_obj.get('job_title', [])
                
#                 # Generate search queries
#                 for location in locations:
#                     for business in businesses:
#                         for job_title in job_titles:
#                             search_query = f"{job_title} {business} {location} site:linkedin.com/in"
#                             results = self.search_linkedin_profiles(search_query, num_results=3)
                            
#                             # Extract URLs
#                             for result in results:
#                                 all_linkedin_urls.append(result['link'])
                            
#                             time.sleep(0.5)  # Rate limiting between searches
            
#             # Remove duplicates
#             unique_urls = list(set(all_linkedin_urls))
#             st.success(f"🎯 Found {len(unique_urls)} unique LinkedIn profiles to enrich")
            
#             if not unique_urls:
#                 return "No LinkedIn profiles found. Try adjusting your search criteria."
            
#             # Step 2: Enrich all profiles with Apify
#             st.subheader("🔍 Step 2: Enriching Profiles with Apify")
#             enriched_profiles = self.batch_enrich_profiles(unique_urls)
            
#             # Count successful enrichments
#             successful_enrichments = len([p for p in enriched_profiles if p.get('scraping_status') == 'success'])
            
#             # Step 3: Save to Google Sheets
#             st.subheader("💾 Step 3: Saving to Google Sheets")
#             save_result = self.save_enriched_data_to_sheets(enriched_profiles)
            
#             # Summary
#             summary = f"""
#             🎉 **Lead Generation Complete!**
            
#             📊 **Results Summary:**
#             - LinkedIn profiles found: {len(unique_urls)}
#             - Successfully enriched: {successful_enrichments}
#             - Failed enrichments: {len(enriched_profiles) - successful_enrichments}
            
#             💾 **Storage:** {save_result}
#             """
            
#             return summary
            
#         except Exception as e:
#             return f"Error in lead generation process: {str(e)}"



class LeadScrapingTool:
    """Tool for scraping leads using Google Custom Search API and enriching with Apify,
    then pushing everything to Google Sheets."""

    # ────────────────────────────────────────────────────────────────────────────
    # INIT
    # ────────────────────────────────────────────────────────────────────────────
    def __init__(self, api_key: str, cse_id: str, sheets_service, apify_token: str):
        self.api_key = api_key
        self.cse_id = cse_id
        self.sheets_service = sheets_service
        self.apify_token = apify_token

    # ────────────────────────────────────────────────────────────────────────────
    # STEP 1 · GOOGLE CUSTOM SEARCH
    # ────────────────────────────────────────────────────────────────────────────
    def search_linkedin_profiles(self, query: str, num_results: int = 12) -> List[Dict]:
        """Search for LinkedIn profile URLs via Google Custom Search."""
        results = []
        start_index = 1

        st.info(f"🔍 Searching for: {query}")

        while len(results) < num_results:
            try:
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    "key": self.api_key,
                    "cx": self.cse_id,
                    "q": query,
                    "start": start_index,
                    "num": min(10, num_results - len(results)),
                }

                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if "items" not in data:
                    break

                for item in data["items"]:
                    if "linkedin.com/in/" in item["link"]:
                        results.append(
                            {
                                "title": item.get("title", ""),
                                "link": item.get("link", ""),
                                "snippet": item.get("snippet", ""),
                                "found_at": datetime.now().isoformat(),
                            }
                        )

                start_index += 10
                time.sleep(0.1)  # CSE rate‑limit

            except Exception as e:
                st.error(f"Error searching: {str(e)}")
                break

        st.success(f"✅ Found {len(results)} LinkedIn profiles")
        return results

    # ────────────────────────────────────────────────────────────────────────────
    # STEP 2 · APIFY ENRICHMENT
    # ────────────────────────────────────────────────────────────────────────────
    def enrich_profile_with_apify(self, linkedin_url: str) -> Dict:
        """Enrich a single LinkedIn profile using Apify."""
        try:
            url = (
                "https://api.apify.com/v2/acts/dev_fusion~linkedin-profile-scraper/"
                "run-sync-get-dataset-items"
            )
            payload = {"profileUrls": [linkedin_url], "maxDelay": 5, "minDelay": 1}
            headers = {
                "Authorization": f"Bearer {self.apify_token}",
                "Content-Type": "application/json",
            }

            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data:
                profile = data[0]
                return {
                    "linkedin_url": linkedin_url,
                    "firstName": profile.get("firstName", ""),
                    "lastName": profile.get("lastName", ""),
                    "fullName": profile.get("fullName", ""),
                    "headline": profile.get("headline", ""),
                    "connections": profile.get("connectionsCount", ""),
                    "followers": profile.get("followersCount", ""),
                    "email": profile.get("email", ""),
                    "mobileNumber": profile.get("mobileNumber", ""),
                    "jobTitle": profile.get("jobTitle", ""),
                    "companyName": profile.get("companyName", ""),
                    "companyIndustry": profile.get("companyIndustry", ""),
                    "companyWebsite": profile.get("companyWebsiteUrl", ""),
                    "companyLinkedin": profile.get("companyLinkedinUrl", ""),
                    "companyFoundedIn": profile.get("companyFoundedYear", ""),
                    "companySize": profile.get("companySize", ""),
                    "location": profile.get("location", ""),
                    "about": profile.get("about", ""),
                    "experience": json.dumps(profile.get("experience", [])),
                    "education": json.dumps(profile.get("education", [])),
                    "skills": json.dumps(profile.get("skills", [])),
                    "industry": profile.get("industry", ""),
                    "profile_picture": profile.get("profilePicture", ""),
                    "scraped_at": datetime.now().isoformat(),
                    "scraping_status": "success",
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
        """Enrich multiple LinkedIn profiles with a progress bar."""
        enriched, total = [], len(linkedin_urls)
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, url in enumerate(linkedin_urls):
            status_text.text(f"🔍 Enriching profile {i + 1}/{total}: {url}")
            enriched.append(self.enrich_profile_with_apify(url))
            progress_bar.progress((i + 1) / total)

            if i < total - 1:  # Respect Apify rate‑limit
                time.sleep(2)

        status_text.text(f"✅ Completed enriching {total} profiles!")
        return enriched

    # ────────────────────────────────────────────────────────────────────────────
    # STEP 3 · SAVE TO GOOGLE SHEETS
    # ────────────────────────────────────────────────────────────────────────────
    def save_enriched_data_to_sheets(
        self, enriched_data: List[Dict], sheet_name: str = "Sheet1"
    ) -> str:
        """Append enriched data to Google Sheets, avoiding duplicates."""
        try:
            spreadsheet = self.sheets_service.open("PyLeads")
            sheet = spreadsheet.worksheet(sheet_name)

            if not enriched_data:
                return "No enriched data to save."

            df_new = pd.DataFrame(enriched_data).fillna("")
            existing_records = sheet.get_all_records()
            df_existing = pd.DataFrame(existing_records) if existing_records else pd.DataFrame()


            start_row = len(existing_records) + 2  # header row + 1‑based
            for i, row in enumerate(df_new.values.tolist()):
                sheet.insert_row(row, index=start_row + i)

            return f"Successfully appended {len(df_new)} new enriched profiles to Google Sheets!"

        except Exception as e:
            return f"Error appending enriched data to sheets: {str(e)}"

    # ────────────────────────────────────────────────────────────────────────────
    # MASTER PIPELINE
    # ────────────────────────────────────────────────────────────────────────────
    def scrape_leads(self, query_json: str) -> str:
        """Full pipeline: search → enrich → save."""
        try:
            st.info("🚀 Starting lead generation process...")
            query_data = json.loads(query_json)

            all_urls: List[str] = []
            search_meta_by_url: Dict[str, Dict] = {}

            # ── Step 1: Google Search ───────────────────────────────────────────
            st.subheader("📋 Step 1: Searching LinkedIn Profiles")

            for q in query_data:
                for location in q.get("location", []):
                    for business in q.get("business", []):
                        for job_title in q.get("job_title", []):
                            search_query = f"{job_title} {business} {location} site:linkedin.com/in"
                            cse_results = self.search_linkedin_profiles(search_query, num_results=12)

                            for r in cse_results:
                                url = r["link"]
                                all_urls.append(url)
                                search_meta_by_url[url] = {
                                    "google_title": r.get("title", ""),
                                    "google_snippet": r.get("snippet", ""),
                                    "google_found_at": r.get("found_at", ""),
                                }
                            time.sleep(0.5)  # between CSE queries

            unique_urls = list(set(all_urls))
            st.success(f"🎯 Found {len(unique_urls)} unique LinkedIn profiles to enrich")

            if not unique_urls:
                return "No LinkedIn profiles found. Adjust your search criteria."

            # ── Step 2: Apify Enrichment ───────────────────────────────────────
            st.subheader("🔍 Step 2: Enriching Profiles with Apify")
            enriched_profiles = self.batch_enrich_profiles(unique_urls)

            # Merge Google‑search metadata into each enriched profile
            for record in enriched_profiles:
                record.update(search_meta_by_url.get(record["linkedin_url"], {}))

            successful = sum(p.get("scraping_status") == "success" for p in enriched_profiles)

            # ── Step 3: Save to Sheets ─────────────────────────────────────────
            st.subheader("💾 Step 3: Saving to Google Sheets")
            sheet_msg = self.save_enriched_data_to_sheets(enriched_profiles)

            return (
                "🎉 **Lead Generation Complete!**\n\n"
                "📊 **Results Summary:**\n"
                f"- LinkedIn profiles found: {len(unique_urls)}\n"
                f"- Successfully enriched: {successful}\n"
                f"- Failed enrichments: {len(enriched_profiles) - successful}\n\n"
                f"💾 **Storage:** {sheet_msg}"
            )

        except Exception as e:
            return f"Error in lead generation process: {str(e)}"


class LeadResearchTool:
    """Tool for individual lead research (optional - for single profile research)"""
    
    def __init__(self, api_token: str, sheets_service):
        self.api_token = api_token
        self.sheets_service = sheets_service
        
    def research_single_profile(self, linkedin_url: str) -> Dict:
        """Research a single LinkedIn profile using Apify"""
        try:
            # Apify API call
            url = "https://api.apify.com/v2/acts/dev_fusion~linkedin-profile-scraper/run-sync-get-dataset-items"
            
            payload = {
                "profileUrls": [linkedin_url],
                "maxDelay": 5,
                "minDelay": 1
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data and len(data) > 0:
                profile = data[0]
                return {
                    'linkedin_url': linkedin_url,
                    'firstName': profile.get('firstName', ''),
                    'lastName': profile.get('lastName', ''),
                    'fullName': profile.get('fullName', ''),
                    'headline': profile.get('headline', ''),
                    'connections': profile.get('connectionsCount', ''),
                    'followers': profile.get('followersCount', ''),
                    'email': profile.get('email', ''),
                    'mobileNumber': profile.get('mobileNumber', ''),
                    'jobTitle': profile.get('jobTitle', ''),
                    'companyName': profile.get('companyName', ''),
                    'companyIndustry': profile.get('companyIndustry', ''),
                    'companyWebsite': profile.get('companyWebsiteUrl', ''),
                    'companyLinkedin': profile.get('companyLinkedinUrl', ''),
                    'companyFoundedIn': profile.get('companyFoundedYear', ''),
                    'companySize': profile.get('companySize', ''),
                    'location': profile.get('location', ''),
                    'about': profile.get('about', ''),
                    'experience': json.dumps(profile.get('experience', [])),
                    'education': json.dumps(profile.get('education', [])),
                    'skills': json.dumps(profile.get('skills', [])),
                    'industry': profile.get('industry', ''),
                    'profile_picture': profile.get('profilePicture', ''),
                    'researched_at': datetime.now().isoformat(),
                    'scraping_status': 'success'
                }
            else:
                return {'error': 'No profile data found'}
                
        except Exception as e:
            return {'error': f'Error researching profile: {str(e)}'}
    
    def save_single_research_to_sheets(self, data: Dict, sheet_name: str = "Individual Research") -> str:
        """Save single research data to Google Sheets"""
        try:
            # Create or open research sheet
            try:
                spreadsheet = self.sheets_service.open("Lead Generation Results")
            except:
                spreadsheet = self.sheets_service.create("Lead Generation Results")
                
            try:
                sheet = spreadsheet.worksheet(sheet_name)
            except:
                sheet = spreadsheet.add_worksheet(title=sheet_name, rows="1000", cols="20")
            
            # Get existing data
            try:
                existing_data = sheet.get_all_records()
                df = pd.DataFrame(existing_data) if existing_data else pd.DataFrame()
            except:
                df = pd.DataFrame()
            
            # Add new research data
            new_df = pd.DataFrame([data])
            df = pd.concat([df, new_df], ignore_index=True)
            
            # Update sheet
            sheet.clear()
            df = df.fillna('')
            data_to_upload = [df.columns.values.tolist()] + df.values.tolist()
            sheet.update(data_to_upload)
            
            return "Individual research data saved successfully!"
            
        except Exception as e:
            return f"Error saving individual research: {str(e)}"
    
    def research_lead(self, linkedin_url: str) -> str:
        """Main research function for single profiles"""
        if not linkedin_url or 'linkedin.com/in/' not in linkedin_url:
            return "Please provide a valid LinkedIn URL (must contain 'linkedin.com/in/')"
        
        st.info(f"🔍 Researching individual profile: {linkedin_url}")
        
        profile_data = self.research_single_profile(linkedin_url)
        
        if 'error' in profile_data:
            return profile_data['error']
        
        save_result = self.save_single_research_to_sheets(profile_data)
        
        return f"✅ Successfully researched profile for {profile_data.get('fullName', 'Unknown')} at {profile_data.get('companyName', 'Unknown Company')}. {save_result}"

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

def create_lead_agent():
    """Create the lead generation agent"""
    
    # Setup services
    sheets_service = setup_google_sheets()
    if not sheets_service:
        return None
    
    # Create tools - pass apify_token to scraping tool
    scraping_tool = LeadScrapingTool(GOOGLE_API_KEY, GOOGLE_CSE_ID, sheets_service, APIFY_API_TOKEN)
    research_tool = LeadResearchTool(APIFY_API_TOKEN, sheets_service)
    
    tools = [
        Tool(
            name="leadScraping",
            description="Use this tool to scrape leads into a Google Sheet. Only call this tool once you have enough information to complete the desired JSON search query.",
            func=scraping_tool.scrape_leads
        ),
        Tool(
            name="leadResearch", 
            description="Use this tool to research a lead by their LinkedIn URL.",
            func=research_tool.research_lead
        )
    ]
    
    # Updated agent prompt based on your requirements
    prompt = PromptTemplate.from_template("""
# Overview
You are a lead generation agent, responsible for scraping and researching leads.

# Tools
### leadScraping:
Use this tool to scrape leads into a Google Sheet. Only call this tool once you have enough information to complete the desired JSON search query.

### leadResearch:
Use this tool to research a lead by their LinkedIn URL.

# Rules
- Ask clarifying questions if you're unsure about something.
- Ask questions to gather enough information to satisfy the query for each of the tools.
- You should introduce yourself as "Lead Generation Joe" and ask the user what leads they want to scrape today.
- Always replace spaces with '+' in your search queries. For example, instead of "los angeles united states", your query should be "los+angeles+united+states".
- You always need a LinkedIn URL to research a lead.
- You can only research one person at a time.
- Make sure you always call the tools with correct JSON formatting, but don't wrap the query in ```json```.
- You should only call the 'leadScraping' tool once per request.
                                      

# Example
- Input: "Hi"
- Output: "Hi, I'm Lead Generation Joe, which leads can I help you scrape today? Just tell me the locations, Business and job titles and let me handle the rest! "
- Input: "Let's do Locations: - Chicago United States - Sydney Australia Business: - Financial Planners"
- Output: "Awesome. I think you forgot the job titles. Which job titles should I search for?"
- Input: "Only CEOs please"
- Call leadScraping tool with query:
[
  {{
    "location": [
      "chicago+united+states",
      "sydney+australia"
    ],
    "business": [
      "financial+planners"
    ],
    "job_title": [
      "ceo"
    ]
  }}
]


You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: ## Response format guidelines
- If you need to call a tool, output exactly:
  Thought: <your thought>
  Action: <tool name>
  Action Input: <tool input>
- **If you are replying to the user directly, DO NOT output an Action block at all.**
  Instead output:
  Thought: <your thought>
  Final Answer: <your answer>
- You should only call a tool **once per request**.
- Once you observe a result from the tool, that will be the final answer.


NEVER write 'Action: None' or 'Action: null'.
Question: {input}
Thought: {agent_scratchpad}
""")
    
    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        base_url="https://openrouter.ai/api/v1/",
    )
    
    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        # callbacks=[StreamlitCallbackHandler(st.container())]
    )
    
    return agent_executor

def main():
    st.set_page_config(
        page_title="Lead Generation System",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Lead Generation System")
    st.markdown("Powered by AI Agent + Web Scraping + Data Enrichment")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key checks
        api_status = {
            "Google API": bool(GOOGLE_API_KEY),
            "OpenAI API": bool(OPENAI_API_KEY),
            "Apify API": bool(APIFY_API_TOKEN),
            "Google Sheets": bool(GOOGLE_SHEETS_CREDENTIALS)
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
        st.markdown("   - 🤖 Enrich EVERY profile with Apify scraper")
        st.markdown("   - 💾 Save all enriched data to Google Sheets")
        st.markdown("4. Optionally research individual profiles with URLs")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = create_lead_agent()
    
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
                        # Fixed: Use invoke instead of run
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

if __name__ == "__main__":
    main()