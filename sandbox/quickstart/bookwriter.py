from dotenv import load_dotenv
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate



load_dotenv()
llm = ChatOpenAI(temperature=0.5)
tools = load_tools(["serpapi"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

def fetch_search_results(topic, agent):
    search_query = f"Find relevant information about {topic}."
    search_results = agent.run(search_query)
    print(search_results)
    return search_results

topic = "The History of the Internet"
# search_results = fetch_search_results(topic, agent)

outline_prompt = PromptTemplate(input_variables=["topic"], template="Write a list of 10 chapters for a book about {topic}. return the list as a string of comma seperated values.")

test_chain = LLMChain(llm=llm, prompt=outline_prompt)
table_of_content = test_chain.run({"topic": topic})



# Create LLMChains for generating chapter titles, section headings, and content paragraphs
# chapter_title_prompt = PromptTemplate(input_variables=["topic"], template="Write a chapter title for a book about {topic}.")
section_heading_prompt = PromptTemplate(input_variables=["topic"], template="Write 6 section headings for a chapter about {topic}. return the list as a string of comma seperated values.")
content_paragraph_prompt = PromptTemplate(input_variables=["topic"], template="Write an approx 300 word paragraph about {topic} ")

# chapter_title_chain = LLMChain(llm=llm, prompt=chapter_title_prompt)

section_heading_chain = LLMChain(llm=llm, prompt=section_heading_prompt)
content_paragraph_chain = LLMChain(llm=llm, prompt=content_paragraph_prompt)

# simple_chain = SimpleSequentialChain(llm=llm,)

for chapter in table_of_content.split(","):
    print(chapter)
    heading = section_heading_chain.run({"topic": chapter})
    for section in heading.split(","):
        print(section)
        paragraphs = content_paragraph_chain.run({"topic": section})
        print(paragraphs)


