from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, function_tool
import os 
from dotenv import load_dotenv
import asyncio
from openai.types.responses import ResponseTextDeltaEvent
load_dotenv()


MODEL_NAME = "gemini-2.0-flash"
API_KEY = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model=MODEL_NAME,
    openai_client=external_client
)


@function_tool
def get_networth_of_person(name:str) -> str:
    """
    Retrieve the net worth of a person by their name.
    """
    mock_data = {
        "Elon Musk": 260_000,
        "Jeff Bezos": 190_000,
        "Taylor Swift": 1_100,
        "Mark Zuckerberg": 170_000,
    }
    # return f"{mock_data.keys} has a net worth of {mock_data.values} million dollars."
    net_worth_value = mock_data.get(name)
    if net_worth_value is not None:
        return f"{name} has a net worth of {net_worth_value} million dollars."
    else:
        return f"Net worth data for {name} is not available."



@function_tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Convert the curreny from one to another based on the given amount and the converted rate is based on USD 
    """
    currency_rate = {
        "euro" : 0.86, 
        "pound" : 0.75,
        "yen" : 147,
        "pkr" : 282,
        "inr": 87.76,
        "usd": 1.0
    }

    mock_coversion_rate = currency_rate.get(to_currency.lower(), 1.0)
    if from_currency.lower() not in currency_rate:
        return f"Conversion from {from_currency} is not supported."
    converted_amount = amount * mock_coversion_rate / currency_rate[from_currency.lower()]
    return f"{amount} {from_currency} is equal to {converted_amount:.2f} {to_currency}."


geographical_agent = Agent(
    name="Geographical Agent",
    instructions="You will answer questions about geographical locations.",
    model=model,
)
health_Agent = Agent(
    name="Health Agent",
    instructions="You will answer questions about health and wellness.",
    model=model,
)
software_agent = Agent(
    name="Software Agent",
    instructions="You will answer questions about software development or any computer related issues.",
    model=model,
)
mechanical_agent = Agent(
    name="Mechanical Agent",
    instructions="You will answer questions about mechanical engineering or any mechanical related issues.",
    model=model,
)
fitness_agent = Agent(
    name="Fitness Agent",
    instructions="You will answer questions about fitness and exercise.",
    model=model,
)
currency_agent = Agent(
    name="Currency Agent",
    instructions="You will answer questions about currency exchange rates.",
    model=model,
)
translator_agent = Agent(
    name="Translator Agent",
    instructions="You will translate languages as per user request.",
    model=model,
)


async def main():
    main_agent = Agent(
        name="Assistant",
        instructions="You will answer questions of the user. if needed then call the tool or handoffs otherwise answer the question.",
        model=model,
        tools=[get_networth_of_person, currency_converter],
        handoffs=[software_agent, geographical_agent, health_Agent, mechanical_agent, fitness_agent, currency_agent, translator_agent]
    )
    
    while True:
        querry = input("Enter Your Querry: ")
        if querry.lower() in ["exit", "quit", "q"]:
            print("Exiting the program.")
            break
        
        result = Runner.run_streamed(
            starting_agent=main_agent,
            input= querry,
        )
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta , end="", flush=True)
    
        

    

if __name__ == "__main__":
    asyncio.run(main())
