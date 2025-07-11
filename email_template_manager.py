from supabase import create_client
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import logging
from langchain_openai import ChatOpenAI
import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class EmailTemplateManager:
    def __init__(self):
        """Initialize the email template manager with Supabase and embedding model."""
        # Initialize Supabase client
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not all([self.supabase_url, self.supabase_key]):
            raise ValueError("Missing required Supabase environment variables")
        
        self.supabase = create_client(self.supabase_url, self.supabase_key)
        
        # Initialize local embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
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

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text using sentence-transformers."""
        return self.embedding_model.encode(text)

    async def retrieve_templates(self, persona: str, stage: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant email drafts to use as templates based on persona and stage.
        
        Args:
            persona: Target persona (e.g., "operations_manager", "facility_manager")
            stage: Email stage (e.g., "initial_outreach", "follow_up")
            top_k: Number of drafts to retrieve
            
        Returns:
            List of relevant drafts with their similarity scores
        """
        try:
            # First, try to get exact matches for persona and stage
            exact_matches = await self.get_templates(persona, stage)
            
            if exact_matches:
                logger.info(f"Found {len(exact_matches)} exact template matches for {persona}/{stage}")
                # Add perfect similarity score to exact matches
                for match in exact_matches:
                    match['similarity'] = 1.0
                return exact_matches
                
            logger.info(f"No exact template matches for {persona}/{stage}, using similarity search")
            
            # If no exact matches, get all templates
            all_templates = await self.get_templates()
            
            # If still no templates, fall back to all drafts
            if not all_templates:
                logger.info("No templates found, falling back to all drafts")
                all_drafts = await self.get_drafts()
                if not all_drafts:
                    logger.warning("No drafts found in the system")
                    return []
                templates_to_search = all_drafts
            else:
                templates_to_search = all_templates

            # Generate embedding for the query
            query = f"{persona} {stage}"
            query_embedding = self.get_embedding(query)
            
            # For each draft, calculate similarity with the query
            drafts_with_scores = []
            for draft in templates_to_search:
                # Generate embedding for the draft content
                draft_content = f"{draft.get('subject', '')} {draft.get('body', '')}"
                if not draft_content.strip():
                    continue
                    
                draft_embedding = self.get_embedding(draft_content)
                
                # Calculate cosine similarity
                similarity = self._calculate_similarity(query_embedding, draft_embedding)
                
                drafts_with_scores.append({
                    **draft,
                    'similarity': float(similarity)  # Convert to float for JSON serialization
                })
            
            # Sort by similarity and get top_k
            sorted_drafts = sorted(drafts_with_scores, key=lambda x: x['similarity'], reverse=True)
            return sorted_drafts[:top_k]
            
        except Exception as e:
            logger.error(f"Error retrieving drafts as templates: {str(e)}")
            return []

    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    async def generate_email(self, lead: Dict[str, Any], templates: List[Dict]) -> Dict[str, str]:
        """
        Generate a personalized email using retrieved drafts as templates and lead information.
        
        Args:
            lead: Dictionary containing lead information
            templates: List of relevant drafts to use as templates
            
        Returns:
            Dictionary containing generated subject and body
        """
        try:
            # Construct the prompt
            system_prompt = """You are an expert email copywriter. Your task is to generate a personalized cold email using the provided successful drafts as inspiration. The email should:
1. Be conversational and natural
2. Reference specific details about the lead
3. Focus on value proposition
4. End with a soft call to action
5. Keep paragraphs short (2-3 sentences)
6. Learn from the style and structure of the provided drafts

Output format must be valid JSON:
{
    "subject": "Your subject line",
    "body": "Your email body"
}"""

            # Format drafts for the prompt, including their similarity scores
            template_examples = "\n\n".join([
                f"Draft {i+1} (Similarity: {t.get('similarity', 0):.2f}):\nSubject: {t.get('subject', '')}\nBody: {t.get('body', '')}"
                for i, t in enumerate(templates)
            ])
            
            # Format lead info with more detail
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
                "content": f"Here are some successful email drafts to learn from (sorted by relevance):\n\n{template_examples}\n\nGenerate a personalized email for this lead:\n{lead_info}\n\nNote: Use the most similar drafts as primary inspiration but adapt to this specific lead."
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
        """
        Save an email draft to Supabase.
        
        Args:
            lead_id: Unique identifier for the lead
            subject: Email subject
            body: Email body
            persona: Target persona (e.g., "operations_manager")
            stage: Email stage (e.g., "initial_outreach")
            
        Returns:
            Dictionary containing operation status and details
        """
        try:
            response = self.supabase.table('email_drafts').insert({
                'lead_id': lead_id,
                'subject': subject,
                'body': body,
                'status': 'draft',
                'persona': persona,
                'stage': stage,
                'created_at': 'now()'
            }).execute()
            
            if not response.data:
                logger.error("No data returned from draft insertion")
                return {
                    "status": "error",
                    "message": "Failed to save draft - no data returned"
                }
                
            logger.info(f"Draft saved successfully with ID: {response.data[0]['id']}")
            return {
                "status": "success",
                "data": response.data[0]
            }
            
        except Exception as e:
            logger.error(f"Error saving draft: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def get_drafts(self, lead_id: str = None, status: str = None) -> List[Dict]:
        """
        Retrieve drafts based on filters.
        
        Args:
            lead_id: Optional - Unique identifier for the lead
            status: Optional - Filter by status (draft, template, sent)
            
        Returns:
            List of draft emails
        """
        try:
            query = self.supabase.table('email_drafts').select('*')
            
            # Apply filters if provided
            if lead_id:
                query = query.eq('lead_id', lead_id)
            
            if status:
                query = query.eq('status', status)
                
            # Execute the query
            response = query.order('created_at', desc=True).execute()
            
            if not response.data:
                logger.info(f"No drafts found with filters: lead_id={lead_id}, status={status}")
                return []
                
            logger.info(f"Found {len(response.data)} drafts")
            return response.data
            
        except Exception as e:
            logger.error(f"Error retrieving drafts: {str(e)}")
            return []
            
    async def get_templates(self, persona: str = None, stage: str = None) -> List[Dict]:
        """
        Retrieve drafts marked as templates, optionally filtered by persona and stage.
        
        Args:
            persona: Optional - Filter by target persona
            stage: Optional - Filter by email stage
            
        Returns:
            List of template drafts
        """
        try:
            # Start with basic query for templates
            query = self.supabase.table('email_drafts').select('*').eq('status', 'template')
            
            # Apply additional filters if provided
            if persona:
                query = query.eq('persona', persona)
                
            if stage:
                query = query.eq('stage', stage)
                
            # Execute the query
            response = query.execute()
            
            return response.data or []
            
        except Exception as e:
            logger.error(f"Error retrieving templates: {str(e)}")
            return []
            
    async def mark_as_template(self, draft_id: str, persona: str, stage: str) -> Dict[str, Any]:
        """
        Mark a draft as a template for future use.
        
        Args:
            draft_id: The ID of the draft to mark as template
            persona: Target persona for this template
            stage: Email stage for this template
            
        Returns:
            Dictionary containing operation status and details
        """
        try:
            response = self.supabase.table('email_drafts').update({
                'status': 'template',
                'persona': persona,
                'stage': stage,
                'updated_at': 'now()'
            }).eq('id', draft_id).execute()
            
            if not response.data:
                logger.error(f"No data returned when marking draft {draft_id} as template")
                return {
                    "status": "error",
                    "message": "Failed to update draft status"
                }
                
            logger.info(f"Draft {draft_id} successfully marked as template")
            return {
                "status": "success",
                "data": response.data[0]
            }
            
        except Exception as e:
            logger.error(f"Error marking draft as template: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
            
    async def mark_as_sent(self, draft_id: str) -> Dict[str, Any]:
        """
        Mark a draft as sent.
        
        Args:
            draft_id: The ID of the draft to mark as sent
            
        Returns:
            Dictionary containing operation status and details
        """
        try:
            response = self.supabase.table('email_drafts').update({
                'status': 'sent',
                'updated_at': 'now()'
            }).eq('id', draft_id).execute()
            
            if not response.data:
                logger.error(f"No data returned when marking draft {draft_id} as sent")
                return {
                    "status": "error",
                    "message": "Failed to update draft status"
                }
                
            logger.info(f"Draft {draft_id} successfully marked as sent")
            return {
                "status": "success",
                "data": response.data[0]
            }
            
        except Exception as e:
            logger.error(f"Error marking draft as sent: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

# Example usage
async def main():
    manager = EmailTemplateManager()
    
    # Example lead
    lead = {
        "fullName": "John Doe",
        "jobTitle": "Operations Manager",
        "companyName": "Acme Corp",
        "companyIndustry": "Manufacturing",
        "location": "New York, USA"
    }
    
    # Get relevant templates
    templates = await manager.retrieve_templates("operations_manager", "initial_outreach")
    
    # Generate email
    result = await manager.generate_email(lead, templates)
    
    if result["status"] == "success":
        # Save as draft
        draft_result = await manager.save_draft(
            lead_id="example_lead_id",
            subject=result["subject"],
            body=result["body"]
        )
        print(f"Draft saved: {draft_result}")
    else:
        print(f"Error: {result['message']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 