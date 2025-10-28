import asyncio

class MockLLM:
    
    def __init__(self, name="MockLLM"):
        self.name = name
        
    async def __call__(self, prompt):
        print(f"\n--- {self.name} Prompt ---")
        print(prompt)
        print(f"\n--- Please provide your response (enter an empty line to finish) ---")
        
        lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        
        return "\n".join(lines)


async def test_mock_llm():
    mock_llm = MockLLM(name="TestLLM")
    
    prompts = [
        "What is the capital of France?",
        "Write a short poem about artificial intelligence."
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nTest {i}:")
        response = await mock_llm(prompt)
        print("\nYour response was:")
        print("-" * 40)
        print(response)
        print("-" * 40)
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_mock_llm())
