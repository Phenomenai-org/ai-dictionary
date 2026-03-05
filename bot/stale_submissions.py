#!/usr/bin/env python3
"""
Stale submission manager for the AI Dictionary.

Runs daily via GitHub Actions. Manages unrevised submissions:
  - 7 days with `needs-revision` and no revision → add `stale` label + reminder
  - 7 days with `stale` label → close issue with final comment
"""

import os
from datetime import datetime, timezone, timedelta

import requests

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
REPO = os.environ.get("GITHUB_REPOSITORY", "donjguido/ai-dictionary")

STALE_WARN_DAYS = 7
STALE_CLOSE_DAYS = 7

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


def get_labeled_issues(label: str, state: str = "open") -> list[dict]:
    """Fetch issues with a given label and the community-submission label."""
    url = f"https://api.github.com/repos/{REPO}/issues"
    params = {
        "labels": f"{label},community-submission",
        "state": state,
        "per_page": 100,
    }
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_label_applied_date(issue_number: int, label_name: str) -> datetime | None:
    """Find when a label was most recently applied to an issue via the timeline API."""
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}/timeline"
    headers = {**HEADERS, "Accept": "application/vnd.github.mockingbird-preview+json"}
    resp = requests.get(url, headers=headers, params={"per_page": 100}, timeout=30)
    if resp.status_code != 200:
        return None

    applied_at = None
    for event in resp.json():
        if (
            event.get("event") == "labeled"
            and event.get("label", {}).get("name") == label_name
        ):
            applied_at = datetime.fromisoformat(
                event["created_at"].replace("Z", "+00:00")
            )

    return applied_at


def comment_on_issue(issue_number: int, body: str):
    """Post a comment on an issue."""
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}/comments"
    requests.post(url, headers=HEADERS, json={"body": body}, timeout=30)


def add_label(issue_number: int, label: str):
    """Add a label to an issue."""
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}/labels"
    resp = requests.post(url, headers=HEADERS, json={"labels": [label]}, timeout=30)
    if resp.status_code == 422:
        create_url = f"https://api.github.com/repos/{REPO}/labels"
        requests.post(
            create_url, headers=HEADERS,
            json={"name": label, "color": "c5def5"}, timeout=30,
        )
        requests.post(url, headers=HEADERS, json={"labels": [label]}, timeout=30)


def close_issue(issue_number: int):
    """Close an issue."""
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}"
    requests.patch(url, headers=HEADERS, json={"state": "closed"}, timeout=30)


def main():
    now = datetime.now(timezone.utc)

    # ── Phase 1: Mark stale issues ────────────────────────────────────────
    print("Checking for issues needing stale warning...")
    needs_revision_issues = get_labeled_issues("needs-revision")

    for issue in needs_revision_issues:
        number = issue["number"]
        labels = [l["name"] for l in issue.get("labels", [])]

        if "stale" in labels:
            continue

        applied = get_label_applied_date(number, "needs-revision")
        if not applied:
            continue

        age = now - applied
        if age >= timedelta(days=STALE_WARN_DAYS):
            print(f"  Issue #{number}: needs-revision for {age.days} days → marking stale")
            add_label(number, "stale")
            comment_on_issue(
                number,
                "This submission has been waiting for revision for "
                f"{age.days} days. It will be closed in {STALE_CLOSE_DAYS} days "
                "if no revision is submitted.\n\n"
                "To revise, post a comment starting with `## Revised Submission` "
                "followed by your updated `### Term`, `### Definition`, etc."
            )

    # ── Phase 2: Close stale issues ───────────────────────────────────────
    print("Checking for stale issues to close...")
    stale_issues = get_labeled_issues("stale")

    for issue in stale_issues:
        number = issue["number"]

        applied = get_label_applied_date(number, "stale")
        if not applied:
            continue

        age = now - applied
        if age >= timedelta(days=STALE_CLOSE_DAYS):
            print(f"  Issue #{number}: stale for {age.days} days → closing")
            comment_on_issue(
                number,
                "Closing due to inactivity. You can still revise by posting a "
                "`## Revised Submission` comment — the bot will reopen and re-evaluate."
            )
            close_issue(number)

    print("Done.")


if __name__ == "__main__":
    main()
