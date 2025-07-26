from supabase import create_client
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import logging
from langchain_openai import ChatOpenAI
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SimpleEmailManager:
    """Simplified email template manager without heavy ML dependencies"""
    
    def __init__(self):
        """Initialize the simple email manager with Supabase only."""
        # Initialize Supabase client
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not all([self.supabase_url, self.supabase_key]):
            raise ValueError("Missing required Supabase environment variables")
        
        self.supabase = create_client(self.supabase_url, self.supabase_key)
        
        # Initialize OpenAI for chat only
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Missing OpenAI API key")
        
        self.llm = ChatOpenAI(
            model="openai/gpt-4.1-mini",
            temperature=0.7,
            openai_api_key=openai_api_key,
            base_url="https://openrouter.ai/api/v1/"
        )

    async def retrieve_templates(self, persona: str, stage: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant email templates based on persona and stage (simplified version)."""
        try:
            # Get exact matches for persona and stage
            exact_matches = await self.get_templates(persona, stage)
            
            if exact_matches:
                logger.info(f"Found {len(exact_matches)} exact template matches for {persona}/{stage}")
                return exact_matches
                
            logger.info(f"No exact template matches for {persona}/{stage}, getting general templates")
            
            # If no exact matches, get general templates
            all_templates = await self.get_templates()
            return all_templates[:top_k] if all_templates else []
            
        except Exception as e:
            logger.error(f"Error retrieving templates: {str(e)}")
            return []

    async def get_templates(self, persona: str = None, stage: str = None) -> List[Dict]:
        """Get email templates from Supabase."""
        try:
            query = self.supabase.table("email_templates").select("*")
            
            if persona:
                query = query.eq("persona", persona)
            if stage:
                query = query.eq("stage", stage)
                
            response = query.execute()
            return response.data if response.data else []
            
        except Exception as e:
            logger.error(f"Error getting templates: {str(e)}")
            return []

    async def get_drafts(self, lead_id: str = None) -> List[Dict]:
        """Get email drafts from Supabase."""
        try:
            query = self.supabase.table("email_drafts").select("*")
            
            if lead_id:
                query = query.eq("lead_id", lead_id)
                
            response = query.execute()
            return response.data if response.data else []
            
        except Exception as e:
            logger.error(f"Error getting drafts: {str(e)}")
            return []

    async def generate_email(self, lead: Dict[str, Any], templates: List[Dict]) -> Dict[str, str]:
        """Generate a personalized email using templates and lead information."""
        try:
            # Construct the prompt
            system_prompt = """You are an expert email copywriter. Your task is to generate a personalized cold email using the provided templates as inspiration. The email should:
1. Be conversational and natural
2. Reference specific details about the lead
3. Focus on value proposition
4. End with a soft call to action
5. Keep paragraphs short (2-3 sentences)
6. Learn from the style and structure of the provided templates

Output format must be valid JSON:
{
    "subject": "Your subject line",
    "body": "Your email body"
}"""

            # Format templates for the prompt
            template_examples = "\n\n".join([
                f"Template {i+1}:\nSubject: {t.get('subject', '')}\nBody: {t.get('body', '')}"
                for i, t in enumerate(templates)
            ])
            
            # Format lead info
            lead_info = f"""
Lead Information:
- Name: {lead.get('fullName', '')}
- Title: {lead.get('jobTitle', '')}
- Company: {lead.get('companyName', '')}
- Industry: {lead.get('companyIndustry', '')}
- Location: {lead.get('location', '')}
- Company Size: {lead.get('companySize', '')}
- LinkedIn: {lead.get('linkedin_url', '')}
- Company Website: {lead.get('companyWebsite', '')}
"""

            # Generate email
            response = await self.llm.ainvoke([{
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": f"Here are some email templates to learn from:\n\n{template_examples}\n\nGenerate a personalized email for this lead:\n{lead_info}"
            }])
            
            try:
                result = json.loads(response.content)
                return {
                    "subject": result["subject"],
                    "body": result["body"],
                    "status": "success"
                }
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM response as JSON")
                return {
                    "status": "error",
                    "message": "Invalid response format from LLM"
                }
                
        except Exception as e:
            logger.error(f"Error generating email: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def save_draft(self, lead_id: str, subject: str, body: str, 
                     persona: str = None, stage: str = None) -> Dict[str, Any]:
        """Save an email draft to Supabase."""
        try:
            draft_data = {
                "lead_id": lead_id,
                "subject": subject,
                "body": body,
                "persona": persona,
                "stage": stage
            }
            
            response = self.supabase.table("email_drafts").insert(draft_data).execute()
            
            if response.data:
                logger.info(f"Draft saved successfully with ID: {response.data[0]['id']}")
                return {
                    "status": "success",
                    "data": response.data[0],
                    "message": "Draft saved successfully"
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to save draft"
                }
                
        except Exception as e:
            logger.error(f"Error saving draft: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def mark_as_template(self, draft_id: str, persona: str, stage: str) -> Dict[str, Any]:
        """Mark a draft as a template."""
        try:
            # First get the draft
            draft_response = self.supabase.table("email_drafts").select("*").eq("id", draft_id).execute()
            
            if not draft_response.data:
                return {
                    "status": "error",
                    "message": "Draft not found"
                }
            
            draft = draft_response.data[0]
            
            # Create template entry
            template_data = {
                "subject": draft["subject"],
                "body": draft["body"],
                "persona": persona,
                "stage": stage
            }
            
            response = self.supabase.table("email_templates").insert(template_data).execute()
            
            if response.data:
                # Note: is_template column may not exist in schema, skipping update
                
                logger.info(f"Template created successfully with ID: {response.data[0]['id']}")
                return {
                    "status": "success",
                    "data": response.data[0],
                    "message": "Template created successfully"
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to create template"
                }
                
        except Exception as e:
            logger.error(f"Error creating template: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def mark_as_sent(self, draft_id: str) -> Dict[str, Any]:
        """Mark a draft as sent."""
        try:
            response = self.supabase.table("email_drafts").update({
                "sent_at": "now()"
            }).eq("id", draft_id).execute()
            
            if response.data:
                logger.info(f"Draft {draft_id} marked as sent")
                return {
                    "status": "success",
                    "message": "Draft marked as sent"
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to mark draft as sent"
                }
                
        except Exception as e:
            logger.error(f"Error marking draft as sent: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
