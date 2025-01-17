from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.youtube_tools import YouTubeTools
# import openai 
# import os
from dotenv import  load_dotenv



load_dotenv()

# openai.api_Key=os.getenv("OPENAI_API_KEY")

## web search agents 
web_search_agent=Agent(
    name="Web search Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.2-3b-preview"),
    tools=[DuckDuckGo()],
    instruction=["Always include sources"],
    show_tools_call=True,
    markdown=True,

)


## financial agent

finance_agent=Agent(
    name="Finanace AI agent",
    model=Groq(id="llama-3.2-3b-preview"),
    tools=[YFinanceTools(stock_price=True,company_info=True,analyst_recommendations=True,stock_fundamentals=True,company_news=True)],
    show_tools_call=True,
    instruction=["use tables to display the data"],
    markdown=True,


)

## YouTube agent
youtube_agent = Agent(
    name="YouTube Agent",
    role="Search YouTube for videos",
    model=Groq(id="llama-3.2-3b-preview"),
    tools=[YouTubeTools()],
    instruction=["Provide video links and summaries"],
    show_tools_call=True,
    markdown=True,
)


##multi ai agent

multi_agent= Agent(
    team=[web_search_agent,finance_agent],
    model=Groq(id="llama-3.2-3b-preview"),
    instruction=["always include sources", "use tables to display data", "add youtube links to view related information"],
    show_tools_call=True,
    markdown=True,
)

response = multi_agent.print_response("share stock price for NVIDIA ", stream=True)
print(response)
