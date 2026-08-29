from app.llm.llm_client import LLMClient


def main():

    print("\n==============================================")
    print("LLM CONNECTION TEST")
    print("==============================================")

    llm = LLMClient()

    prompt = """
You are an exam answer evaluator.

Question:
What is Software Engineering?

Model Answer:
Software Engineering is the systematic and disciplined
approach to the development, operation, testing, and
maintenance of software.

Student Answer:
Software engineering is a systematic approach used to
develop, test and maintain software.

Evaluate the student answer.

Give:
1. Marks out of 5
2. Short justification
3. Missing concepts
"""

    print("\nSending prompt to LLM...")

    result = llm.generate(prompt)

    print("\n==============================================")
    print("LLM RESPONSE")
    print("==============================================")

    print(result)


if __name__ == "__main__":
    main()
