import requests
from github import Github
import datetime
import json

# GitHub token for authentication
GITHUB_TOKEN = ""
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

# Repositories
REPOS = ["autowarefoundation/autoware"]

# Date filter (commits, issues, pull requests created on or before Dec 1, 2024)
DATE_FILTER = "2024-12-01T23:59:59Z"

def fetch_issues_and_prs(repo, entity_type):
    """
    Fetch issues or pull requests for a repository using the GitHub API.
    """
    url = f"https://api.github.com/repos/{repo}/{entity_type}"
    params = {
        "state": "closed",
        "per_page": 100,
        "sort": "created",
        "direction": "desc"
    }
    data = []
    while url:
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code != 200:
            print(f"Error fetching {entity_type} for {repo}: {response.status_code}")
            break
        page_data = response.json()
        for item in page_data:
            if item["created_at"] <= DATE_FILTER:
                data.append(item)
        # Pagination
        url = response.links.get("next", {}).get("url")
    return data

def fetch_commits(repo):
    """
    Fetch commits for a repository using the GitHub API.
    """
    url = f"https://api.github.com/repos/{repo}/commits"
    params = {
        "per_page": 100,
        "until": DATE_FILTER
    }
    commits = []
    while url:
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code != 200:
            print(f"Error fetching commits for {repo}: {response.status_code}")
            break
        page_data = response.json()
        commits.extend(page_data)
        # Pagination
        url = response.links.get("next", {}).get("url")
    return commits

def identify_defect_fixes(pull_requests):
    """
    Identify pull requests that fix defects (merged with defect-related keywords).
    """
    defect_keywords = ["fix", "bug", "defect", "issue", "error", "resolve"]
    defect_fixes = []
    for pr in pull_requests:
        # Safely retrieve title and body
        title = pr.get("title") or ""  # Default to an empty string if None
        body = pr.get("body") or ""    # Default to an empty string if None
        
        # Check if merged and contains defect-related keywords
        if pr.get("merged_at") and any(
            keyword in (title + body).lower() for keyword in defect_keywords
        ):
            defect_fixes.append(pr)
    return defect_fixes

def main():
    for repo in REPOS:
        print(f"Processing repository: {repo}")
        # Fetch commits
        commits = fetch_commits(repo)
        print(f"Total commits fetched for {repo}: {len(commits)}")
        
        # Fetch issues
        issues = fetch_issues_and_prs(repo, "issues")
        print(f"Total issues fetched for {repo}: {len(issues)}")
        
        # Fetch pull requests
        pull_requests = fetch_issues_and_prs(repo, "pulls")
        print(f"Total pull requests fetched for {repo}: {len(pull_requests)}")
        
        # Identify defect-fixing pull requests
        defect_fixes = identify_defect_fixes(pull_requests)
        print(f"Defect-fixing pull requests for {repo}: {len(defect_fixes)}")
        
        # Save results to files
        with open(f"{repo.replace('/', '_')}_commits.json", "w") as f:
            f.write(json.dumps(commits, indent=2))
        with open(f"{repo.replace('/', '_')}_issues_new.json", "w") as f:
            f.write(json.dumps(issues, indent=2))
        with open(f"{repo.replace('/', '_')}_defect_fixes.json", "w") as f:
            f.write(json.dumps(defect_fixes, indent=2))

if __name__ == "__main__":
    main()
