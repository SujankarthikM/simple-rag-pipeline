from openai import OpenAI

api_key = 'e0947c6072ba8d6676bb2117dd903ce6db5eb8856731b162cddc2993cfba99a5'
base_url = 'https://aiplatform.dev51.cbf.dev.paypalinc.com/cosmosai/llm/v1'

client = OpenAI(
    api_key = api_key,
    base_url = base_url
    
)

messages = [{"role": "user", "content": "Hi, who are you."}]

models = client.models.list()
print(models)

response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=messages,
    max_tokens=64,
    temperature=0
)
print(f'response: {response.choices[0].message.content.lstrip()}')

