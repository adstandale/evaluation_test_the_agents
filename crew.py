# legal_guide.py
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from acp import send_acp_message
from deep_eval import evaluate_agent_response

load_dotenv(Path(__file__).resolve().with_name(".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")  # Default to a specific model if not set
llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY) if GROQ_API_KEY else None
search_tool = DuckDuckGoSearchRun(
    api_wrapper=DuckDuckGoSearchAPIWrapper(backend="lite", source="text", region="wt-wt")
)


def search_legal_frameworks(query: str) -> str:
    """Searches the web for legal documents, constitution details, and national laws for a given country."""
    try:
        return search_tool.run(query)
    except Exception as exc:
        return (
            f"Search tool failed for query '{query}': {exc}. "
            "Use official government and constitutional sources and general legal knowledge for the summary."
        )


def create_legal_guide(country_name: str, acp_message: dict | None = None) -> str:
    if llm is None:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to the project's .env file before running the legal guide.")

    send_acp_message(
        "crew_agent",
        "main_agent",
        "legal_guide_received",
        {"country_name": country_name, "acp_message": acp_message},
    )

    search_query = (
        f"{country_name} constitution legal system government structure laws legal framework official "
        "government sites"
    )
    search_results = search_legal_frameworks(search_query)

    research_prompt = f"""
    You are a senior legal researcher and analyst.
    Research the legal system of {country_name} using the provided search results.

    Requirements:
    - identify the form of government
    - find the country's constitution or foundational legal document
    - list at least five important legal domains
    - include official or credible sources when possible
    - summarize the government structure and key legal obligations

    Search results:
    {search_results}
    """

    research_response = llm.invoke([
        SystemMessage(content="You are a legal researcher producing a factual and structured legal summary."),
        HumanMessage(content=research_prompt),
    ])

    analysis_prompt = f"""
    Based on the research below, write a clean legal blueprint and compliance guide for {country_name}.
    Format it as markdown with sections:
    1. Executive Summary
    2. Constitutional Foundations
    3. Key Statutes and Legal Domains
    4. Business and Commercial Compliance
    5. Immigration, Residency, and Labor Rules
    6. Local Customs, Traffic, and Daily Compliance Notes
    7. Sources and References

    Research summary:
    {research_response.content}
    """

    final_response = llm.invoke([
        SystemMessage(content="You are a legal analyst and educator. Write a clear, professional, and practical handbook."),
        HumanMessage(content=analysis_prompt),
    ])

    evaluate_agent_response(
        agent_name="legal_guide_agent",
        user_input=f"Legal guide for {country_name}",
        response=final_response.content,
        expected_output=f"A structured legal guide for {country_name} covering the constitution, legal domains, business compliance, immigration, and local obligations.",
        context=[
            f"Country: {country_name}",
            f"Research summary: {research_response.content}",
        ],
    )

    send_acp_message(
        "crew_agent",
        "main_agent",
        "legal_guide_completed",
        {"country_name": country_name, "characters": len(final_response.content)},
    )

    return final_response.content

def target_country(target_country):
    return target_country

if __name__ == "__main__":
    target_country = input("Enter the country name to generate its legal guide: ").strip() or "India"
    
    print(f"\n[!] Starting legal research for: {target_country}...\n")

    legal_handbook = create_legal_guide(target_country)

    print("\n=================== GENERATED LEGAL HANDBOOK ===================\n")
    print(legal_handbook)
