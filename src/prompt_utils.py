"""The utility functions for prompting GPT and Google Cloud models."""

import time
import openai
import os
from typing import List
from transformers import Conversation
from vllm import LLM, SamplingParams

model2model_path = {
    "llama2-chat-7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-chat-13b": "meta-llama/Llama-2-13b-chat-hf",
    "llama2-chat-7b": "meta-llama/Llama-2-7b-hf",
}


def call_openai_server_func(
    prompt, n=1, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8
):
    """The function to call OpenAI server with an input string."""
    try:
        if isinstance(prompt, str):
            completion = openai.ChatCompletion.create(
                model=model,
                n=n,
                temperature=temperature,
                max_tokens=max_decode_steps,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            completions_list = []
            for i in range(n):
                completions_list.append(
                    completion.choices[i].message.content
                )
            # return completion.choices[0].message.content
            return completions_list
        elif isinstance(prompt, list):
            completion = openai.ChatCompletion.create(
                model=model,
                n=n,
                temperature=temperature,
                max_tokens=max_decode_steps,
                messages=prompt,
            )
            completions_list = []
            for i in range(n):
                completions_list.append(
                    completion.choices[i].message.content
                )
            # return completion.choices[0].message.content
            return completions_list

    except openai.error.Timeout as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 5
        print(f"Timeout error occurred. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_func(
            prompt, n=n, max_decode_steps=max_decode_steps, temperature=temperature
        )

    except openai.error.RateLimitError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 5
        print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_func(
            prompt, max_decode_steps=max_decode_steps, temperature=temperature
        )

    except openai.error.APIError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 5
        print(f"API error occurred. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_func(
            prompt, n=n, max_decode_steps=max_decode_steps, temperature=temperature
        )

    except openai.error.APIConnectionError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 5
        print(f"API connection error occurred. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_func(
            prompt, n=n, max_decode_steps=max_decode_steps, temperature=temperature
        )

    except openai.error.ServiceUnavailableError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 5
        print(f"Service unavailable. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_func(
            prompt, n=n, max_decode_steps=max_decode_steps, temperature=temperature
        )

    except openai.error.AuthenticationError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 5
        print(f"Authentication unavailable. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_func(
            prompt, n=n, max_decode_steps=max_decode_steps, temperature=temperature
        )

    except openai.error.PermissionError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 5
        print(f"Permission unavailable. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_func(
            prompt, n=n, max_decode_steps=max_decode_steps, temperature=temperature
        )
        
    except openai.error.InvalidRequestError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 5
        print(f"Invalid request error occurred. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_func(
            prompt, n=n, max_decode_steps=max_decode_steps, temperature=temperature
        )

    except OSError as e:
        retry_time = 5  # Adjust the retry time as needed
        print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_func(
            prompt, n=n, max_decode_steps=max_decode_steps, temperature=temperature
        )

def call_vllm_server_func(
    prompt, model, llm_name, max_decode_steps=20, temperature=0.8, stop_tokens=None
):
    """The function to call vllm with a list of input strings."""

    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=max_decode_steps, stop=stop_tokens
    )
    res_completions = []
    
    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(prompt, list):
        if all(isinstance(elem, str) for elem in prompt):
            completions = model.generate(prompts=prompt, sampling_params=sampling_params)
        elif all(isinstance(sublist, list) and all(isinstance(item, int) for item in sublist) for sublist in prompt):
            completions = model.generate(prompt_token_ids=prompt, sampling_params=sampling_params)

    for output in completions:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        res_completions.append(generated_text)

    return res_completions


def load_vllm_llm(model, tensor_parallel_size):
    llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.9, trust_remote_code=True)
    return llm


if __name__ == "__main__":
    prompt = "The sun rises from the west."
    res_list = call_openai_server_func(prompt, max_decode_steps=50, n=8)
    print(res_list)
    print(res_list[0])
    print(type(res_list[0]))
