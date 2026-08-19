import json
from typing import List, Optional
from pydantic import BaseModel, Field

# 1. Define the Structured Output Schema for validation
class EvaluationResult(BaseModel):
    is_hallucinated: bool = Field(
        ..., 
        description="Set to True if any claim in the AI response cannot be found in the source text."
    )
    unsupported_claims: List[str] = Field(
        default=[], 
        description="A list of specific sentences or facts that were hallucinated or unverified."
    )
    justification: str = Field(
        ..., 
        description="Step-by-step reasoning explaining why the response is grounded or where it fails."
    )

# 2. Mock function simulating an API call to Claude acting as an LLM Judge
def mock_claude_judge_api(source_text: str, ai_response: str) -> str:
    """
    In a real application, you would use the anthropic SDK:
    
    import anthropic
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        temperature=0.0,
        system="You are a strict data validation judge. Output ONLY valid JSON matching the schema.",
        messages=[{"role": "user", "content": f"Source: {source_text}\n\nEvaluate: {ai_response}"}]
    )
    return message.content[0].text
    """
    # Mocking a detected hallucination response from Claude
    mock_json_output = {
        "is_hallucinated": True,
        "unsupported_claims": [
            "The model launched in Germany in November 2024."
        ],
        "justification": "The source text states that the model launched in France and the UK in October 2024. It makes no mention of Germany or November 2024."
    }
    return json.dumps(mock_json_output)

# 3. Main execution script for validation
def validate_ai_output():
    # Setup test data
    source_context = "Anthropic released the new model in France and the UK in October 2024 for enterprise testing."
    
    # This response contains a hallucination (Germany/November)
    unverified_ai_response = "The model launched in Germany in November 2024 for enterprise testing."
    
    print("--- Starting Output Validation ---")
    print(f"Source Context: {source_context}")
    print(f"AI Response to Check: {unverified_ai_response}\n")
    
    try:
        # Get raw text response from the judging model
        raw_judge_response = mock_claude_judge_api(source_context, unverified_ai_response)
        
        # Parse and validate the response using Pydantic
        # This guarantees structural compliance before using the data in production code
        parsed_result = EvaluationResult.model_validate_json(raw_judge_response)
        
        # Act based on validation results
        if parsed_result.is_hallucinated:
            print("❌ VALIDATION FAILED: Hallucination Detected!")
            print(f"Justification: {parsed_result.justification}")
            print(f"Flagged Claims: {parsed_result.unsupported_claims}")
            # Here you would trigger an alert, block the output, or retry the prompt
        else:
            print("✅ VALIDATION PASSED: Output is fully grounded in the source text.")
            
    except Exception as e:
        print(f"Formatting Error: The judge failed to return valid JSON matching the schema. Error: {e}")

if __name__ == "__main__":
    validate_ai_output()
