"""
publish_to_hf.py
Publishes Tenacious-Bench v0.1 to HuggingFace Datasets.

Usage:
    python publish_to_hf.py --username your-hf-username

What gets uploaded:
    - tenacious_bench_v0.1/train/   (125 tasks)
    - tenacious_bench_v0.1/dev/     (75 tasks)
    - README.md, datasheet.md, schema_tenacious_bench.json
    - examples/ directory

What does NOT get uploaded:
    - tenacious_bench_v0.1/held_out/  (embargoed per datasheet)
    - training_data/pairs.jsonl        (training data, not benchmark)
    - .env, .env.example               (secrets)
"""

import argparse
import sys
from pathlib import Path


def add_dataset_card_frontmatter(readme_path: Path) -> str:
    """Prepend HuggingFace dataset card YAML to README content."""
    frontmatter = """---
license: cc-by-4.0
task_categories:
  - text-generation
language:
  - en
tags:
  - benchmark
  - b2b-sales
  - evaluation
  - simpo
  - lora
  - preference-learning
size_categories:
  - n<1K
pretty_name: Tenacious-Bench v0.1
---

"""
    content = readme_path.read_text(encoding="utf-8")
    if content.startswith("---"):
        return content  # already has frontmatter
    return frontmatter + content


def main():
    parser = argparse.ArgumentParser(description="Publish Tenacious-Bench to HuggingFace")
    parser.add_argument("--username", required=True, help="Your HuggingFace username")
    parser.add_argument("--repo-name", default="tenacious-bench", help="Dataset repo name")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without uploading")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    REPO_ID = f"{args.username}/{args.repo_name}"
    print(f"Target: https://huggingface.co/datasets/{REPO_ID}")

    if args.dry_run:
        print("\n[DRY RUN] Would upload:")
        for p in ["tenacious_bench_v0.1/train", "tenacious_bench_v0.1/dev",
                  "README.md", "datasheet.md", "schema_tenacious_bench.json", "examples/"]:
            print(f"  {p}")
        print("\nRun without --dry-run to publish.")
        return

    api = HfApi()

    # Create repo if it doesn't exist
    print(f"\nCreating dataset repo: {REPO_ID}")
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )
        print("Repo ready.")
    except Exception as e:
        print(f"ERROR creating repo: {e}")
        print("Make sure you ran: hf auth login")
        sys.exit(1)

    # Write README with frontmatter to a temp location
    readme_path = Path("README.md")
    if readme_path.exists():
        readme_content = add_dataset_card_frontmatter(readme_path)
        tmp_readme = Path("_README_hf.md")
        tmp_readme.write_text(readme_content, encoding="utf-8")
    else:
        tmp_readme = None
        print("WARNING: README.md not found")

    # Upload train split
    train_dir = Path("tenacious_bench_v0.1/train")
    if train_dir.exists():
        print(f"\nUploading train split ({len(list(train_dir.glob('*.json')))} tasks)...")
        api.upload_folder(
            folder_path=str(train_dir),
            repo_id=REPO_ID,
            repo_type="dataset",
            path_in_repo="data/train",
        )
        print("Train split uploaded.")
    else:
        print("WARNING: tenacious_bench_v0.1/train not found")

    # Upload dev split
    dev_dir = Path("tenacious_bench_v0.1/dev")
    if dev_dir.exists():
        print(f"\nUploading dev split ({len(list(dev_dir.glob('*.json')))} tasks)...")
        api.upload_folder(
            folder_path=str(dev_dir),
            repo_id=REPO_ID,
            repo_type="dataset",
            path_in_repo="data/dev",
        )
        print("Dev split uploaded.")
    else:
        print("WARNING: tenacious_bench_v0.1/dev not found")

    # Upload examples
    examples_dir = Path("examples")
    if examples_dir.exists():
        print("\nUploading examples...")
        api.upload_folder(
            folder_path=str(examples_dir),
            repo_id=REPO_ID,
            repo_type="dataset",
            path_in_repo="examples",
        )
        print("Examples uploaded.")

    # Upload documentation files
    doc_files = {
        "_README_hf.md" if tmp_readme else None: "README.md",
        "datasheet.md": "datasheet.md",
        "schema_tenacious_bench.json": "schema_tenacious_bench.json",
    }
    print("\nUploading documentation...")
    for local, remote in doc_files.items():
        if local and Path(local).exists():
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=remote,
                repo_id=REPO_ID,
                repo_type="dataset",
            )
            print(f"  {remote} uploaded.")

    # Clean up temp README
    if tmp_readme and tmp_readme.exists():
        tmp_readme.unlink()

    print(f"\n{'='*60}")
    print(f"Published successfully!")
    print(f"Dataset URL: https://huggingface.co/datasets/{REPO_ID}")
    print(f"{'='*60}")
    print("\nNOTE: held_out/ was NOT uploaded (embargoed per datasheet.md)")
    print("Upload it manually after final evaluation completes.")


if __name__ == "__main__":
    main()
