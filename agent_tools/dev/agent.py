from openai import OpenAI

client = OpenAI()
context = []

def call():
    return client.responses.create(model="gpt-5-mini", input=context)

def process(line):
    context.append({"role": "user", "content": line})
    response = call()
    context.append({"role": "assistant", "content": response.output_text})
    return response.output_text

def main():
    try:
        while True:
            line = input("$$; ")
            result = process(line)
            print(f"%%; {result}\n")
    except KeyboardInterrupt:
        print(f"Here's the final context string: \n {context}\n")

if __name__ == "__main__": main()

