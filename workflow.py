import os
import sys
import json
from anthropic import Anthropic

# Initialize the client. It automatically picks up os.environ.get("ANTHROPIC_API_KEY")
client = Anthropic()

def run_workflow_pipeline(customer_email: str) -> dict:
    """
    Executes a 2-step solution design pattern:
    Step 1: Classify email intent (Using Claude 3.5 Haiku for speed/cost efficiency)
    Step 2: Generate specialized response (Using Claude 3.5 Sonnet for reasoning quality)
    """
    print("🚀 Step 1: Analyzing and Classifying Email...")
    
    # SYSTEM PROMPT: Forces Claude to output ONLY valid JSON using pre-filling techniques
    classification_prompt = (
        "You are an automated email classifier. Analyze the customer email and classify it "
        "into one of these categories: 'TECHNICAL', 'BILLING', or 'SALES'. "
        "Provide a brief 1-sentence reason for your choice.\n\n"
        f"Customer Email:\n\"\"\"\n{customer_email}\n\"\"\"\n\n"
        "Output your response strictly as a JSON object with keys 'category' and 'reason'."
    )

    try:
        # Step 1 Call (Optimized with Haiku)
        classification_response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=150,
            temperature=0.0,  # Factual, deterministic sorting
            messages=[{"role": "user", "content": classification_prompt}]
        )
        
        # Parse the output
        raw_json = classification_response.content[0].text.strip()
        metadata = json.loads(raw_json)
        category = metadata.get("category", "TECHNICAL")
        reason = metadata.get("reason", "")
        
        print(f"✅ Classified as: {category}")
        print(f"   Reason: {reason}\n")

    except Exception as e:
        print(f"❌ Step 1 Error: Failed to parse classification metadata. {e}")
        sys.exit(1)

    # Dynamic Routing Logic based on Step 1 Output
    prompt_routing = {
        "TECHNICAL": "You are a senior support engineer. Provide step-by-step troubleshooting instructions.",
        "BILLING": "You are a billing representative. Address their payment issue politely and outline next steps.",
        "SALES": "You are an account executive. Acknowledge their interest and offer to schedule a discovery call."
    }
    
    selected_system_role = prompt_routing.get(category, prompt_routing["TECHNICAL"])
    
    print(f"🚀 Step 2: Routing to {category} Expert Model...")

    # Step 2 Call (Optimized with Sonnet for high-quality text output)
    generation_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        temperature=0.5,  # Slight creativity for conversational warmth
        system=selected_system_role,
        messages=[
            {"role": "user", "content": f"Draft a professional reply to this email:\n{customer_email}"}
        ]
    )
    
    ai_draft = generation_response.content[0].text
    
    # Construct final consolidated pipeline payload
    return {
        "pipeline_metadata": {
            "assigned_routing": category,
            "routing_justification": reason
        },
        "final_draft": ai_draft
    }

# --- Execution Example ---
if __name__ == "__main__":
    # Test Scenario: A complex billing dispute
    sample_email = (
        "Hello, I am writing because I noticed an overcharge of $45 on my invoice #INV-9821. "
        "I upgraded my plan last week, but the promotion code 'SAVE50' wasn't applied correctly. "
        "Please credit my account before the next auto-renew cycle."
    )
    
    print("--- Starting AI Solution Workflow Pipeline ---\n")
    pipeline_result = run_workflow_pipeline(sample_email)
    
    print("--- Final Solution Outputs (Ready for Human Review) ---")
    print(f"Routing metadata: {json.dumps(pipeline_result['pipeline_metadata'], indent=2)}")
    print("\nDraft Response:")
    print(pipeline_result['final_draft'])
