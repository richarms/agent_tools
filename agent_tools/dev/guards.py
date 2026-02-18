import random
from openai import OpenAI
client = OpenAI()

context_good, context_bad = [{
    "role": "system", "content": "you're Alph and you only tell the truth"
}], [{
    "role": "system", "content": "you're Ralph and you only tell lies"
}]

def call(ctx):
    return client.responses.create(model="gpt-5.1-mini", input=ctx)

def process(line):
    context_good.append({"role": "user", "content": line})
    context_bad.append({"role": "user", "content": line})
    if random.choice([True, False]):
        response = call(context_good)
    else:
        response = call(context_bad)        
    context_good.append({"role": "assistant", "content": response.output_text})        
    context_bad.append({"role": "assistant", "content": response.output_text})        
    return response.output_text

def main():
    try:
        while True:
            line = input("$$; ")
            result = process(line)
            print(f"%%; {result}\n")
    except KeyboardInterrupt:
        print("Here's the final truth (Alph) context string:")
        print(context_good)
        print("Here's the final lies (Ralph) context string:")
        print(context_bad)

if __name__ == "__main__": main()
