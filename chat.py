from openai import OpenAI
import os
import argparse

client = OpenAI(
    base_url="https://localhost/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)


def main():
    parser = argparse.ArgumentParser(description="Query OpenAI but wuth mitmproxy that blocks suspicious prompts")
    parser.add_argument("prompt", type=str, help="Prompt to send to the model")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name like gpt-4o")

    args = parser.parse_args()
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "user", "content": args.prompt}
        ]
    )    
    if (response.choices) :
        print(response.choices[0].message.content.strip())
    else:
        print(response.message)   

if __name__ == "__main__":
    main()