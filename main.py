import os
from typing import TypedDict, Annotated
import operator

import psycopg
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import(
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,   
)

from deep_eval import evaluate_agent_response

from langchain_groq import ChatGroq
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from dotenv import load_dotenv
from acp import get_trace_id, send_acp_message
from crew import create_legal_guide

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
)

DATABASE_URL=os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Add it to your .env file before running the app.")


def safe_print_text(value: str) -> str:
    return value.encode("ascii", "replace").decode("ascii")


class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    country_name: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
    
def flight_agent(state: TravelState):
    query=state["user_query"]
    flight_data=search_flights(query)
    
    return {
         "flight_results": flight_data,
         "messages":[
             AIMessage(content=f"Flight search results for query '{query}':\n{flight_data}")
             
         ],
         "llm_calls": state.get("llm_calls", 0) + 1       
}
    
def hotel_agent(state: TravelState):
    query=f"Best hotels for {state['user_query']}"
    hotel_results=tavily_search(query)
    
    return{
        "hotel_results": hotel_results,
        "messages":[
            AIMessage(content=f"Hotel search results for query '{query}':\n{hotel_results}")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1    
        
    }
    
def itinerary_agent(state: TravelState):
    prompt=f"""
    create a travel itinerary.
    User query: {state['user_query']}
    
    Flight results: {state['flight_results']}
    
    Hotel results: {state['hotel_results']}
    """
    
    response=llm.invoke([
        SystemMessage(
            content="You are a travel assistant that creates travel itineraries based on user queries and search results."
        ),
        HumanMessage(content=prompt)
        
    ])
    evaluate_agent_response(
        agent_name="itinerary_agent",
        user_input=state.get("user_query", ""),
        response=response.content,
        expected_output="A clear, structured travel itinerary that matches the user's trip and the flight/hotel search context.",
        context=[
            f"User query: {state.get('user_query', '')}",
            f"Flight results: {state.get('flight_results', '')}",
            f"Hotel results: {state.get('hotel_results', '')}",
        ],
    )
    return{
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1 
    }
    
def final_agent(state: TravelState):
    final_prompt=f"""
    Generate a final travel itinerary based on the following information:
    
    Flights: {state['flight_results']}
    Hotels: {state['hotel_results']}
    Itinerary: {state['itinerary']}
    """
    response=llm.invoke([
        HumanMessage(content=final_prompt)
    ])
    evaluate_agent_response(
        agent_name="final_agent",
        user_input=state.get("user_query", ""),
        response=response.content,
        expected_output="A final travel response that integrates flights, hotel information, and the itinerary into a usable recommendation.",
        context=[
            f"Flights: {state.get('flight_results', '')}",
            f"Hotels: {state.get('hotel_results', '')}",
            f"Itinerary: {state.get('itinerary', '')}",
        ],
    )
    
    return{
        "messages":[response],        
        "llm_calls": state.get("llm_calls", 0) + 1
    }


def legal_guide_agent(state: TravelState):
    country_name = state.get("country_name") or state.get("user_query", "India")
    acp_request = send_acp_message(
        "main_agent",
        "crew_agent",
        "legal_guide_request",
        {"country_name": country_name, "user_query": state.get("user_query", ""), "trace_id": get_trace_id()},
    )
    legal_guide = create_legal_guide(country_name, acp_message=acp_request)
    return {
        "messages": [AIMessage(content=f"Legal guide for '{country_name}':\n{legal_guide}")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }
    
graph = StateGraph(TravelState)    

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)
graph.add_node("legal_guide_agent", legal_guide_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", "legal_guide_agent")
graph.add_edge("legal_guide_agent", END)

_conn=psycopg.connect(DATABASE_URL, autocommit=True)
checkpointer=PostgresSaver(_conn)
checkpointer.setup()

app=graph.compile(checkpointer=checkpointer)


if __name__=="__main__":
    config={
        "configurable": {
            "thread_id": "user_anandprakash",
        }
    }

    country_name = input("Enter the country name: ").strip() or "India"
    user_input = input(f"Enter your travel query for {country_name}: ")

    trace_id = get_trace_id()
    send_acp_message(
        "main_agent",
        "crew_agent",
        "travel_request",
        {"country_name": country_name, "user_query": user_input, "trace_id": trace_id},
    )

    legal_guide = create_legal_guide(country_name)
    print("\n\nLEGAL GUIDE FOR:", country_name)
    print(safe_print_text(legal_guide))

    result=app.invoke(
        {
        "messages":[
            HumanMessage(content=user_input)
        ],
        "user_query": user_input,
        "country_name": country_name,
        "flight_results": "",
        "hotel_results": "",
        "itinerary": "",
        "llm_calls": 0
        },
        config=config
    )
    
    print("\n\nFINAL RESPONSE\n\n")
    
    for msg in result["messages"]:
        print(safe_print_text(msg.content))
        