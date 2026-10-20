import os
from pathlib import Path
from openai import OpenAI

# Initialize standard OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Strict prompt optimized to minimize reasoning steps and conversational filler
LOW_TOKEN_SYSTEM_PROMPT = """
You are a concise Code Auditor. Review the provided multi-file project source code against the Functional Requirements.
Perform a structural and semantic analysis to verify if the code accurately implements the requirements.

Provide your output strictly in this markdown format:
### 📊 Requirements Checklist
- [REQ_ID]: [PASS/FAIL] - Brief explanation.

### 🐛 Critical Flaws & Logic Deviations
- File `filename.py`: Description of logic error or missing requirement implementation.

### 💡 Direct Recommendations
- Bulleted actionable code fixes.
Do not include introductory or concluding conversational filler text.
"""

def bundle_project_structure(project_dir: str) -> str:
    """Combines all project code into a highly compact text block to save tokens."""
    excluded_dirs = {'.git', '__pycache__', '.venv', 'env', 'node_modules', '.vscode'}
    allowed_extensions = {'.py', '.json', '.yaml', '.yml'}
    
    bundled_text = ""
    
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix in allowed_extensions and file_path.name != "low_token_reviewer.py":
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    # Strip empty lines and simple comment lines to reduce input tokens
                    clean_lines = [line for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]
                    compact_content = "\n".join(clean_lines)
                    
                    # Wrap cleanly so model understands the file boundaries
                    bundled_text += f"\n--- FILE: {file_path.name} ---\n{compact_content}\n"
                except Exception as e:
                    pass
                    
    return bundled_text

def run_low_token_review(project_dir: str, requirements_file: str):
    print("📉 Bundling files for a low-token static review...")
    
    # 1. Read requirements
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements_content = f.read()
        
    # 2. Bundle and minify workspace source code
    project_code_bundle = bundle_project_structure(project_dir)
    
    # 3. Use standard single-shot ChatCompletion (No Assistant, No Threads, No Sandbox)
    response = client.chat.completions.create(
        model="gpt-4o-mini", # 💡 Use gpt-4o-mini here! It is ~20x cheaper than gpt-4o and excellent at static audits
        messages=[
            {"role": "system", "content": LOW_TOKEN_SYSTEM_PROMPT},
            {"role": "user", "content": f"REQUIREMENTS:\n{requirements_content}\n\nPROJECT CODE:\n{project_code_bundle}"}
        ],
        temperature=0.1 # Low temperature ensures deterministic, analytical results
    )
    
    print("\n================📋 STATIC REVIEW REPORT 📋================")
    print(response.choices[0].message.content)
    
    # Print usage statistics to monitor token efficiency
    print("\n================📉 TOKEN CONSUMPTION 📉================")
    print(f"Input Tokens: {response.usage.prompt_tokens}")
    print(f"Output Tokens: {response.usage.completion_tokens}")
    print(f"Total Tokens: {response.usage.total_tokens}")

if __name__ == "__main__":
    run_low_token_review("./src", "./requirements.md")
