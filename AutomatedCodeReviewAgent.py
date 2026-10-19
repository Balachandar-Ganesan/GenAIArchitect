import os
import time
from pathlib import Path
from openai import OpenAI

# Initialize the OpenAI Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Strict role definition for handling complex, multi-file code hierarchies
AGENT_INSTRUCTIONS = """
You are an expert Principal QA Automation Engineer and Software Auditor. 
You are evaluating a multi-file software project against specified Functional Requirements.

You have access to the actual workspace files uploaded to your active directory.
Your objective is to:
1. **Understand Architecture:** Map out how the uploaded files interact and import modules from one another.
2. **Review:** Evaluate if the combined logic fulfills the Functional Requirements document completely.
3. **Execute Integration Tests:** Write a thorough test suite file (e.g., `test_suite.py`) inside the code interpreter container. This suite must import the components from the uploaded project files and validate negative paths, positive paths, and requirement edge-cases.
4. **Report:** Provide a highly structured summary outlining which requirements passed or failed, the terminal test output logs, and precise architectural or logical fixes if any error triggers.
"""

def run_workspace_review(project_dir: str, requirements_file: str):
    print(f"🔍 Scanning VSCode project directory: {project_dir}")
    
    # 1. Locate the Functional Requirements Document
    req_path = Path(requirements_file)
    if not req_path.exists():
        print(f"❌ Requirements file not found at: {requirements_file}")
        return
    with open(req_path, "r", encoding="utf-8") as f:
        requirements_content = f.read()

    # 2. Automatically discover local project source files to upload
    # (Excluding virtual environments, caches, and test artifacts to save tokens)
    excluded_dirs = {'.git', '__pycache__', '.venv', 'env', 'node_modules', '.vscode'}
    allowed_extensions = {'.py', '.json', '.yaml', '.yml', '.txt'}
    
    uploaded_file_ids = []
    file_mapping_notes = []

    for root, dirs, files in os.walk(project_dir):
        # Filter out excluded directories in-place
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix in allowed_extensions and file_path.name != "review_project.py":
                print(f"📤 Uploading file to Sandbox: {file_path.name}")
                
                # Upload file reference securely to OpenAI
                with open(file_path, "rb") as f_data:
                    uploaded_file = client.files.create(
                        file=f_data,
                        purpose="assistants"
                    )
                uploaded_file_ids.append(uploaded_file.id)
                file_mapping_notes.append(f"- File `{file_path.name}` uploaded with ID: {uploaded_file.id}")

    if not uploaded_file_ids:
        print("⚠️ No valid source files discovered to upload.")
        return

    # 3. Create the Ephemeral Assistant Agent with Code Interpreter enabled
    assistant = client.beta.assistants.create(
        name="Local Project Test Agent",
        instructions=AGENT_INSTRUCTIONS,
        model="gpt-4o",  # Advanced reasoning model required for multi-file graph mapping
        tools=[{"type": "code_interpreter"}]
    )

    # 4. Bind the uploaded workspace file IDs directly to the Thread's Code Interpreter resource
    # This mounts your actual file entities directly into the runtime context of the container
    thread = client.beta.threads.create(
        tool_resources={
            "code_interpreter": {
                "file_ids": uploaded_file_ids
            }
        }
    )

    # 5. Build your targeted query payload referencing the mapped environment files
    file_mapping_str = "\n".join(file_mapping_notes)
    user_payload = f"""
Please evaluate the multi-file project files uploaded to your runtime environment.

### TARGET FUNCTIONAL REQUIREMENTS:
{requirements_content}

### ENVIRONMENT FILES MAP:
{file_mapping_str}

Please dynamically inspect these files, map their mutual dependencies/imports, generate a Python integration testing module, execute it via your code interpreter shell, and return the architectural test logs.
"""

    # 6. Post prompt message to conversation execution thread
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_payload
    )

    print("🤖 Agent analyzes multi-file repository structure and executing integration tests...")
    
    # 7. Start the active run pipeline
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    # 8. Actively poll run cycle until termination
    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status == 'completed':
            print("✅ Multi-file workspace validation complete!")
            break
        elif run_status.status in ['failed', 'cancelled', 'expired']:
            print(f"❌ Sandbox pipeline run failed: {run_status.status}")
            return
        time.sleep(3)

    # 9. Extract and render generated structural evaluation response
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    final_response = messages.data[0].content[0].text.value

    # 10. File & Tool Cleanup Lifecycle
    print("🧹 Cleaning up remote sandbox workspace files and tools...")
    client.beta.assistants.delete(assistant.id)
    for f_id in uploaded_file_ids:
        client.files.delete(f_id)

    print("\n================📋 PROJECT ARCHITECTURE REVIEW REPORT 📋================")
    print(final_response)


if __name__ == "__main__":
    # Ensure your paths align with your local VSCode workspace setup
    # Assuming your workspace has a 'requirements.md' file and a source subfolder 'src' (or project root '.')
    PROJECT_WORKSPACE_PATH = "./src"  # Points to your module folder
    REQUIREMENTS_DOC_PATH = "./requirements.md"
    
    run_workspace_review(PROJECT_WORKSPACE_PATH, REQUIREMENTS_DOC_PATH)
