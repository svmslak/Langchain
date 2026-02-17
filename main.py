from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama



load_dotenv()



def main():

    print("Hello from helloworld!")

    information = """India is a country in South Asia. It is the seventh-largest country by land area 
    and the second-most populous country in the world, with over 1.3 billion people. India has a rich history 
    and diverse culture, with many different languages, religions, and traditions. The capital of India is New Delhi,
    and the largest city is Mumbai."""

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    prompt = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    #llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm = ChatOllama(model="llama3.2-vision:11b", temperature=0)
    chain = prompt | llm
    result = chain.invoke(input={"information": information})
    print(result)

if __name__ == "__main__":
    main()
