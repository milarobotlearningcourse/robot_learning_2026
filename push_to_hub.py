import os
from huggingface_hub import HfApi, login, HfFolder

# Path to the model and config
MODEL_PATH = "/home/ben/Documents/Projects/ift6163/robot_learning_2026/outputs/2026-01-29/02-34-11/miniGRP.pth"
CONFIG_PATH = "/home/ben/Documents/Projects/ift6163/robot_learning_2026/outputs/2026-01-29/02-34-11/.hydra/config.yaml"
CODE_PATH = "/home/ben/Documents/Projects/ift6163/robot_learning_2026/mini-grp/grp_model.py"

def push_model():
    print("Pushing model to Hugging Face Hub...")
    
    # Check if logged in
    if HfFolder.get_token() is None:
        print("You are not logged in to Hugging Face.")
        print("Please enter your Hugging Face token (found at https://huggingface.co/settings/tokens):")
        login()
    
    repo_id = input("Enter the Hugging Face repository ID (e.g., username/mini-grp): ").strip()
    if not repo_id:
        print("Repository ID is required.")
        return

    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        print(f"Creating/checking repository {repo_id}...")
        api.create_repo(repo_id=repo_id, exist_ok=True)
    except Exception as e:
        print(f"Error accessing repo: {e}")
        return

    # Upload model
    if os.path.exists(MODEL_PATH):
        print(f"Uploading {MODEL_PATH}...")
        api.upload_file(
            path_or_fileobj=MODEL_PATH,
            path_in_repo="miniGRP.pth",
            repo_id=repo_id,
            repo_type="model"
        )
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    
    # Upload config
    if os.path.exists(CONFIG_PATH):
        print(f"Uploading {CONFIG_PATH}...")
        api.upload_file(
            path_or_fileobj=CONFIG_PATH,
            path_in_repo=".hydra/config.yaml",
            repo_id=repo_id,
            repo_type="model"
        )
    else:
        print(f"Error: Config file not found at {CONFIG_PATH}")
        return

    if os.path.exists(CODE_PATH):
        print(f"Uploading {CODE_PATH}...")
        api.upload_file(
            path_or_fileobj=CODE_PATH,
            path_in_repo="grp_model.py",
            repo_id=repo_id,
            repo_type="model"
        )
    else:
        print(f"Error: Code file not found at {CODE_PATH}")
        return
    
    print(f"Upload complete! View your model at https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    push_model()
