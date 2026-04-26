import argparse
from huggingface_hub import HfApi
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="HuggingFace token")
    args = parser.parse_args()

    # The repo ID string using the username given
    repo_id = "convaiinnovations/shoeguard-safety-slm-q4_k_m"
    
    # Initialize the API client
    api = HfApi(token=args.token)

    print(f"Creating repository: {repo_id}...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print("Repository created (or already exists).")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    print("Uploading README.md...")
    try:
        api.upload_file(
            path_or_fileobj="model/README.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model"
        )
        print("README.md uploaded successfully.")
    except Exception as e:
        print(f"Error uploading README: {e}")

    model_path = "model/safety-q4_k_m.gguf"
    if os.path.exists(model_path):
        print(f"Uploading model file: {model_path}...")
        try:
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo="safety-q4_k_m.gguf",
                repo_id=repo_id,
                repo_type="model"
            )
            print("Model file uploaded successfully.")
        except Exception as e:
            print(f"Error uploading model file: {e}")
    else:
        print(f"Warning: Model file not found at {model_path}. Skipping model upload. Make sure the file exists.")

    dataset_path = "model/imu_actions_1000.jsonl"
    if os.path.exists(dataset_path):
        print(f"Uploading dataset file: {dataset_path}...")
        try:
            api.upload_file(
                path_or_fileobj=dataset_path,
                path_in_repo="imu_actions_1000.jsonl",
                repo_id=repo_id,
                repo_type="model"
            )
            print("Dataset file uploaded successfully.")
        except Exception as e:
            print(f"Error uploading dataset file: {e}")
    else:
        print(f"Warning: Dataset file not found at {dataset_path}. Skipping dataset upload.")
        
    print("Done!")

if __name__ == "__main__":
    main()