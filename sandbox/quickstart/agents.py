from dotenv import load_dotenv
from langchain.agents import load_tools
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI


load_dotenv()

def write_file_wrapper(input_string):
    filename, content = input_string.split("|", 1)
    with open(filename, "w") as f:
        f.write(content)
    return f"File '{filename}' has been written."

def main():
    # First, let's load the language model we're going to use to control the agent.
    llm = ChatOpenAI(temperature=0.5)

    # Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
    write_file_tool = Tool(
        name="WriteToFile",
        func=write_file_wrapper,
        description="Write content to a file. The input should be in the format 'filename|content'."
    )
    read_file_tool = Tool(
        name="ReadFromFile",
        func=lambda filename: open(filename, "r").read(),
        description="Read content from a file. The input should be the filename."
    )

    tools = load_tools(
        ["serpapi", "wikipedia"],
    ) + [write_file_tool, read_file_tool]

    # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
    agent = initialize_agent(
        tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    prompt = "Write an essay on why gpt is a game changer from a software developers perspective." \
        "Step1 - Create an outline of what you want to write, including chapters and subsections. " \
        "Step2 - Search the web for information on the topic. " \
        "Step3 - Modify the original outline to account for newly found information. " \
        "Step4 - Construct each section of the essay based on the search results, and write each one to its own txt file. " \
        "Step5 - Merge the sections into a single txt file. " \
        "Step6 - Proofread the essay, ensuring it is at least 5000 words long. " \
        "Step7 - Search the web again for additional information if needed." \
        "Step8 - Make modifications to the essay as needed." \
        "Step9 - Repeat steps 6-8 until the essay is at least 5000 words long and coherent. " \
        "Step10 - Write the final essay to a txt file."

    # Now let's test it out!
    agent.run(input=prompt, max_tokens=4096, num_samples=1, save_samples=True
    )


if __name__ == "__main__":
    with get_openai_callback() as cb:
        main()
        print("total_tokens", cb.total_tokens)
        print("total_cost", cb.total_cost)


