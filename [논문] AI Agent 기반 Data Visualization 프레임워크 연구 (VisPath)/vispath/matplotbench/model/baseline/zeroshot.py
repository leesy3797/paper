import asyncio
import re
import os
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential, stop_after_delay,
)

# Load environment variables
load_dotenv()

class ZeroShot:
    def __init__(self, api_key=None, base_url=None, model="gpt-4o-mini", system_prompt=None, temperature=0.2):
        # Use environment variables if no parameters are provided
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("BASE_URL")
        self.model = model
        self.system_prompt = system_prompt or "Based on the user's query, generate Python code using `matplotlib.pyplot` and 'seaborn' to create the requested plot. If data is needed, use data from the data path provided. Ensure the code is outputted within the format ```python...```."
        self.temperature = temperature
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def describe_data(self, data_path):
        # Read the data file using pandas
        # data = pd.read_csv(data_path)
        # description = {
        #     "data_path": data_path,
        #     "columns": list(data.columns),
        #     "dtypes": data.dtypes.to_dict(),
        #     "shape": data.shape,
        # }
        return str(data_path)
    
    # def describe_data(self, data_path):
    #     if not data_path:
    #         return "No data file provided."
    #     try:
    #         data = pd.read_csv(data_path)
    #         description = {
    #             "data_path": data_path,
    #             "columns": list(data.columns),
    #             "dtypes": data.dtypes.to_dict(),
    #             "shape": data.shape,
    #         }
    #         return str(description)
    #     except Exception as e:
    #         return f"Error reading data file: {e}"


    @retry(wait=wait_random_exponential(min=0.02, max=1), stop=(stop_after_delay(3) | stop_after_attempt(30)))
    async def call_openai_api(self, system_prompt, user_content):
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=self.temperature
            )
            
            # Extracting response content and token usage
            message_content = response.choices[0].message.content
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            return {
                "response": message_content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
            
        except Exception as e:
            print(f"API call failed with error: {e}. Retrying...")
            raise e
            
    async def call_openai_api(self, user_query, data_description):
        # Retry indefinitely until successful
        while True:
            try:
                # Concatenate the data description and user query
                # full_query = f"Data Description: {data_description}\nUser Query: {user_query}"
                full_query = f"Data Path: {data_description}\nUser Query: {user_query}"
                
                # print(full_query)
                
                # Call OpenAI API to get the response
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": full_query}
                    ],
                    temperature=self.temperature
                )
                # Extract the code content from the response
                response_text = response.choices[0].message.content
                match = re.search(r'```python\n(.*?)```', response_text, flags=re.DOTALL)
                if match:
                    return match.group(1).strip()
                else:
                    return None
            except Exception as e:
                print(f"API call failed with error: {e}. Retrying...")

    async def get_code_content(self, user_query, data_path_list, img_file_path=None):
        print(f'사용하는 모델 명 : {self.model}')        
        data_description = []
        # data_description = ''
        if data_path_list:
            for data_path in data_path_list:
                root_path = os.path.abspath(os.path.dirname(os.curdir))
                data_path = os.path.normpath(os.path.join(root_path, data_path))
            data_description.append(data_path)
        else:
            data_description =  None
        # Call API and return the generated code
        code_content = await self.call_openai_api(user_query, data_description)
        return code_content