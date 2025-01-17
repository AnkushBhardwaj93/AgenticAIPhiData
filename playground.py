from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import  load_dotenv
from phi.playground import Playground, serve_playground_app
import os
import phi

#Load environment variables .env
load_dotenv()


phi.api=os.getenv("PHI_API_KEY")


## web search agents 
web_search_agent=Agent(
    name="Web search Agent",
    role="Search the web for information",
    model=Groq(id="llama3-70b-8192"),
    tools=[DuckDuckGo()],
    instruction=["Always include sources"],
    show_tools_call=True,
    markdown=True,

)


## financial agent

finance_agent=Agent(
    name="Finanace AI agent",
    model=Groq(id="llama3-70b-8192"),
    tools=[YFinanceTools(stock_price=True,company_info=True,analyst_recommendations=True,stock_fundamentals=True,company_news=True)],
    show_tools_call=True,
    instruction=["use tables to display the data"],
    markdown=True,


)

app = Playground(agents=[finance_agent,web_search_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app",reload=True)
