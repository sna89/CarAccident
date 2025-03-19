import tiktoken

system_message = """
You are an expert at condensing text while preserving key information. 
Your task is to shorten the provided text to contain no more than {target_tokens} tokens 
(currently it has {current_tokens} tokens).

Guidelines:
1. Preserve the core meaning and critical details
2. Remove redundant information and verbose language
3. Keep any specific instructions or requirements
4. Maintain the original request's intent
5. Do not add new information that wasn't in the original text
6. Focus on making the text more concise, not on changing its purpose
"""


def count_tokens(text, model_name):
    encoding = tiktoken.encoding_for_model(model_name)
    token_count = len(encoding.encode(text))
    return token_count


def shorten_prompt(client, prompt, target_tokens, model_name):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Please shorten this text to be under {target_tokens} tokens:\n\n{prompt}"}
        ],
        temperature=0.1,
        max_tokens=min(4096, target_tokens)
    )

    shortened_prompt = response.choices[0].message.content.strip()
    new_token_count = count_tokens(shortened_prompt, model_name)
    print("New token count: {}".format(new_token_count))
    return shortened_prompt
