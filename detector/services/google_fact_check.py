
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = (
    "https://factchecktools.googleapis.com/"
    "v1alpha1/claims:search"
)


def search_fact_checks(query, language="en"):
    """
    Search Google's database for previously fact-checked claims.

    Returns matching fact-check reviews.
    """

    api_key = os.getenv("GOOGLE_FACT_CHECK_API_KEY")

    if not api_key:
        return {
            "success": False,
            "error": "Google Fact Check API key is not configured.",
            "results": []
        }

    if not query or not query.strip():
        return {
            "success": False,
            "error": "Search query cannot be empty.",
            "results": []
        }

    params = {
        "query": query.strip(),
        "languageCode": language,
        "pageSize": 5,
        "key": api_key
    }

    try:
        response = requests.get(
            API_URL,
            params=params,
            timeout=15
        )

        response.raise_for_status()
        data = response.json()

        results = []

        for claim in data.get("claims", []):
            for review in claim.get("claimReview", []):
                publisher = review.get("publisher", {})

                results.append({
                    "claim": claim.get("text", ""),
                    "claimant": claim.get("claimant", ""),
                    "claim_date": claim.get("claimDate", ""),
                    "publisher": publisher.get("name", ""),
                    "publisher_site": publisher.get("site", ""),
                    "title": review.get("title", ""),
                    "rating": review.get("textualRating", ""),
                    "review_date": review.get("reviewDate", ""),
                    "url": review.get("url", ""),
                    "language": review.get("languageCode", "")
                })

        return {
            "success": True,
            "error": None,
            "results": results
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Google Fact Check API request timed out.",
            "results": []
        }

    except requests.exceptions.HTTPError:
        try:
            error_message = response.json().get("error", {}).get(
                "message", "Google API request failed."
            )
        except ValueError:
            error_message = "Google API request failed."

        return {
            "success": False,
            "error": error_message,
            "results": []
        }

    except requests.exceptions.RequestException as error:
        return {
            "success": False,
            "error": str(error),
            "results": []
        }
