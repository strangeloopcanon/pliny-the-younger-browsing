#!/usr/bin/env python3
"""Utility helpers for the Pliny browsing environment."""

from __future__ import annotations

import html
from typing import Optional
from urllib.parse import urlparse, urlunparse


def canonicalize_url(url: str) -> str:
    """Normalize URLs for graph keys.

    - Strips whitespace
    - Removes fragments (#...)
    - Normalizes scheme/host casing
    """
    if not url:
        return ""
    u = url.strip()
    if not u:
        return ""
    parsed = urlparse(u)
    # Drop fragments, keep query (gives deterministic states for same path+query)
    sanitized = parsed._replace(fragment="")
    # Normalize scheme/host to lowercase
    scheme = sanitized.scheme.lower()
    netloc = sanitized.netloc.lower()
    sanitized = sanitized._replace(scheme=scheme, netloc=netloc)
    return urlunparse(sanitized)


def infer_domain(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if not host:
        return "general"
    return host


def infer_page_type(url: str, title: Optional[str] = None) -> str:
    """Approximate page type for CSV-derived pages.

    We favor broad categories using host and path heuristics.
    """
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    if not host:
        return "general"
    if "github.com" in host:
        if "/pull" in path:
            return "github_pull_request"
        if "/issues" in path:
            return "github_issue"
        if path.count("/") <= 2:
            return "github_repository"
        return "github_page"
    if "x.com" in host or "twitter.com" in host:
        return "social_feed"
    if "reddit.com" in host:
        return "social_discussion"
    if "google.com" in host and "search" in path:
        return "search_results"
    if "youtube.com" in host:
        return "video"
    if "docs." in host or "documentation" in path:
        return "documentation"
    if "peacocktv.com" in host or "netflix.com" in host or "hulu.com" in host:
        return "streaming"
    if "stackoverflow.com" in host:
        return "developer_qna"
    if "wikipedia.org" in host or "britannica.com" in host:
        return "reference"
    if "accounts.google.com" in host:
        return "auth"
    if "console.firebase.google.com" in host:
        return "cloud_console"
    # Default: prefix with csv_ so it's distinguishable from template data
    core = host.split(":", 1)[0]
    segment = core.split(".")
    if len(segment) >= 2:
        core = segment[-2]
    return f"csv_{core}"


def clean_title(title: str) -> str:
    if not title:
        return ""
    return html.unescape(title.strip())


def domain_label(url: str, page_type: Optional[str] = None) -> str:
    lowered = (url or "").lower()
    if "github.com" in lowered or (page_type or "").startswith("github"):
        return "github"
    if "vercel.com" in lowered or "deployment" in (page_type or ""):
        return "vercel"
    if "x.com" in lowered or "twitter.com" in lowered:
        return "social"
    if "slack.com" in lowered:
        return "slack"
    if "docs." in lowered or "documentation" in (page_type or ""):
        return "docs"
    if "google.com" in lowered and "search" in lowered:
        return "search"
    if page_type:
        return page_type
    host = urlparse(url).netloc.lower()
    if host:
        parts = host.split(".")
        if len(parts) >= 2:
            return parts[-2]
        return host
    return "general"
