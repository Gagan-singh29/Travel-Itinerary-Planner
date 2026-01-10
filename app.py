import gradio as gr
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

class PlannerState(TypedDict):
    messages : Annotated[
        List[HumanMessage | AIMessage],
          "the messages in the conversation"
          ]
    city: str
    interests: List[str]
    itinerary: str

#Define the llm
llm = ChatGroq(
    temperature = 0,
    groq_api_key = "GROK_API_KEY",
    model_name = "llama-3.3-70b-versatile"
)

#Define the itinerary prompt
itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Create a day trip itinerary for {city} based on the user's interests: {interests}. Provide a brief, bulleted itinerary."),
    ("human", "Create an itinerary for my day trip."),
])

def input_city(state: PlannerState) -> PlannerState:
    print("Please enter the city you want to visit for your day trip: ")
    user_message = input("Your Input: ")
    return {
        **state,
        "city": user_message,
        "messages": state['messages'] + [HumanMessage(content=user_message)]
        }

def input_interest(state: PlannerState) -> PlannerState:
    print(f"Please enter your interest for the trip to : {state['city']} (comma-separted): ")
    user_message = input("Your Input: ")
    return {
        **state,
        "interests": [interest.strip() for interest in user_message.split(",")],
        "messages": state['messages'] + [HumanMessage(content=user_message)]
        }
def create_itinerary(state: PlannerState) -> PlannerState:
    print(f"Creating an itinerary for {state['city']} based on interests : {', '.join(state['interests'])}")
    response = llm.invoke(itinerary_prompt.format_messages(city = state['city'], interests = ','.join(state['interests'])))
    print("\nFinal Itinerary: ")
    print(response.content)
    return {
        **state,
        "messages": state['messages'] + [AIMessage(content=response.content)],
        "itinerary" : response.content
    }

#Define the Gradio Application
def travel_planner(city_input: str, interests_input: str):
    print(f"City: {city_input}, Interests: {interests_input}\n")
    state = {
        "messages": [HumanMessage(content=f"Plan a trip to {city_input} with interests {interests_input}")],
        "city": city_input,
        "interests": [interest.strip() for interest in interests_input.split(",")],
        "itinerary": "",
    }

    # Generate the itinerary using the pre-filled state
    updated_state = create_itinerary(state)

    return updated_state["itinerary"]

# Build the Gradio interface
interface = gr.Interface(
    fn=travel_planner,
    theme= 'Yntec/HaleyCH_Theme_Orange_Green',
    inputs=[
        gr.Textbox(label= "Enter the city for your day trip"),
        gr.Textbox(label= "Enter your interests (comma-separated)"),
    ],
    outputs=gr.Textbox(label="Generated Itinerary", lines=10),
    title="HarryAI",
    description="""
    <div style="text-align:center; margin-top:5px;">
        <h3>Travel Itinerary Planner</h3>
        <p>
            Enter a city and your interests to generate a personalized day trip itinerary.
        </p>
    </div>
    """
)

#Launch the Gradio application
interface.launch()
