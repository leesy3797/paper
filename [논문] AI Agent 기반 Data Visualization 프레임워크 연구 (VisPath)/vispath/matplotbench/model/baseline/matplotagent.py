import asyncio
import re
import os
import pandas as pd
import base64
from openai import AsyncOpenAI
from dotenv import load_dotenv
import io
from copy import deepcopy
# from evaluation.utils import code_to_image
import matplotlib.pyplot as plt
from loguru import logger # type: ignore
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential, stop_after_delay,
)


# Load environment variables
load_dotenv()

QUERY_EXPANSION_PROMPT='''According to the user query, expand and solidify the query into a step by step detailed instruction (or comment) on how to write python code to fulfill the user query's requirements. Import the appropriate libraries. Pinpoint the correct library functions to call and set each parameter in every function call accordingly.'''

SYSTEM_PROMPT_CODE_GEN='''You are a helpful assistant that generates Python code for data visualization and analysis using matplotlib, seaborn and pandas. Given a detailed instruction and data description, generate the appropriate code in the Markdwon format of ```python ...```. MUST FOLLOW THE FORMAT.'''

FEEDBACK_SYSTEM_PROMPT = '''Given a piece of code, a user query, and an image of the current plot, please determine whether the plot has faithfully followed the user query. Your task is to provide instruction to make sure the plot has strictly completed the requirements of the query. Please output a detailed step by step instruction on how to use python code to enhance the plot.'''

FEEDBACK_USER_PROMPT = '''Here is the code: [Code]:
"""
{{code}}
"""

Here is the user query: [Query]:
"""
{{query}}
"""

Carefully read and analyze the user query to understand the specific requirements. Examine the provided Python code to understand how the current plot is generated. Check if the code aligns with the user query in terms of data selection, plot type, and any specific customization. Look at the provided image of the plot. Assess the plot type, the data it represents, labels, titles, colors, and any other visual elements. Compare these elements with the requirements specified in the user query. Note any differences between the user query requirements and the current plot. Based on the identified discrepancies, provide step-by-step instructions on how to modify the Python code to meet the user query requirements. Suggest improvements for better visualization practices, such as clarity, readability, and aesthetics, while ensuring the primary focus is on meeting the user's specified requirements. If there is no base64 image due to an error(EX. "Error during Saving plot image: [Errno 2] No such file or directory: 'your_data.csv"), please check the error message and provide feedback based on the specific issue. The feedback should suggest appropriate actions to resolve the issue according to the error details.
'''

class MatplotAgent:
    def __init__(self, api_key=None, base_url=None, model="gpt-4o-mini", temperature=0.2):
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("BASE_URL")
        self.model = model
        self.temperature = temperature
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
            )

    # async def _call_openai_api(self, system_prompt, user_content):
    #     while True:
    #         try:
    #             response = await self.client.chat.completions.create(
    #                 model=self.model,
    #                 messages=[
    #                     {"role": "system", "content": system_prompt},
    #                     {"role": "user", "content": user_content}
    #                 ],
    #                 max_tokens = 15000,
    #                 temperature=self.temperature
    #             )
    #             return response.choices[0].message.content
    #         except Exception as e:
    #             print(f"API call failed with error: {e}. Retrying...")


    @retry(wait=wait_random_exponential(min=0.02, max=1), stop=(stop_after_delay(3) | stop_after_attempt(30)))
    async def _call_openai_api(self, system_prompt, user_content):
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call failed with error: {e}. Retrying...")
            raise e
        

    async def _query_extension(self, nl_query):
        return await self._call_openai_api(QUERY_EXPANSION_PROMPT, nl_query)
            
    def _describe_data(self, data_path):
        if not data_path:
            return "No data file provided."
        try:
            data = pd.read_csv(data_path)
            description = {
                "data_path": data_path,
                "columns": list(data.columns),
                "dtypes": data.dtypes.apply(lambda x: x.name).to_dict(),
                "shape": data.shape,
                "sample": data.head(3).to_dict()
            }
            return str(description)
        except Exception as e:
            return f"Error reading data file: {e}"

    def add_idx(self, path):
        parts = path.rsplit('.', 1)  
        if len(parts) == 2:
            return ".".join([f"{parts[0]}_before_feedback", parts[1]]) 
        return path  

    def code_to_image2(self, code, img_save_path):
        import matplotlib.pyplot as plt
        exec_globals = {"plt": plt, "io": io}
        exec_locals = {}
        print('Start Executing Code and Save Final Image')
        try:
            code_n = code.replace("plt.show()", f"plt.savefig('{img_save_path}')\nplt.close('all')")
            exec(code_n, exec_globals, exec_locals)
            message = "Save Image Successfully!"
            print(message)
            return code, True, None
        except Exception as e:
            message = f"Error during Save : {str(e)}"
            print(message)
            return code, False, str(e)
         
    async def get_code_content(self, nl_query, data_path_list, img_save_path):
            
        if data_path_list:
            data_description_list = []
            for data_path in data_path_list:
                root_path = os.path.abspath(os.path.dirname(os.curdir))
                data_path = os.path.normpath(os.path.join(root_path, data_path))
                single_data_description = self._describe_data(data_path)
                data_description_list.append(single_data_description)
            data_description = "[" + "], [".join(data_description_list) + "]"

        else :
            data_description='''There is no dataset provided.'''

        extended_query = await self._query_extension(nl_query)
        
        user_content = f"""Detailed Instructions:
{extended_query}

Data Description:
Note that you must use the right csv data which is stated in "data_path". DO NOT MOCK DATA if data_path is provided!!!
If there are no data_path then just follow the Detailed Instructions. There might be an instruction about the data.
{data_description}

Please generate Python code using `matplotlib.pyplot` and 'seaborn' and 'pandas' to create the requested plot. Ensure the code is outputted within the Markdown format like ```python\n...```. 
You MUST follow the format and Don't savefig or save anything else.
"""
        
        try_count = 0
        while try_count < 4:
            print('Getting Code with Max4 trials')
            response_text = await self._call_openai_api(SYSTEM_PROMPT_CODE_GEN, user_content)
            # print(response_text)
            match = re.search(r'```python\n(.*?)```', response_text, flags=re.DOTALL)
            # print(match)
            if match:
                code = match.group(1).strip()
                # print(code)
                tmp_img_path = self.add_idx(img_save_path)
                code, log, error_message = self.code_to_image2(code, tmp_img_path)
                # print(log)
                if log:
                    # print(code)
                    print('Stop Before Max4')
                    return code, tmp_img_path, None, data_description
                else:
                    print(f'Try Again {try_count+1}')
                    try_count += 1
            else:
                print(f'Try Again {try_count+1}')
                try_count += 1
        print('No code unitl Max4')
        return code, None, error_message, data_description

    # Function to encode the image to base64 format
    def encode_image(self, image_path):
        print('Encoding Image to Base64')
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    async def visual_feedback(self, ori_query, code, img_save_path, data_description):
        logger.info('Starting Feedback...')
        # Encode the image to base64 for sending as a message
        try:
            base64_image = self.encode_image(img_save_path)
        except:
            base64_image = "No image found due to Error"
            
        # Prepare the messages for GPT-4o, including the system prompt, user prompt with code and query, and the image
        messages = [
            {"role": "system", 
            "content": '''Given a piece of code, a user query, and an image of the current plot, please determine whether the plot has faithfully followed the user query. Your task is to provide instruction to make sure the plot has strictly completed the requirements of the query. Please output a detailed step by step instruction on how to use python code to enhance the plot.'''},
            {"role": "user", 
            "content": f'''Here is the code: [Code]:
    """
    {code}
    """

    Here is the user query and the about the data description: [Query]:
    """
    {ori_query}

    {data_description}
    """

    Carefully read and analyze the user query to understand the specific requirements. Examine the provided Python code to understand how the current plot is generated. Check if the code aligns with the user query in terms of data selection, plot type, and any specific customization. Look at the provided image of the plot. Assess the plot type, the data it represents, labels, titles, colors, and any other visual elements. Compare these elements with the requirements specified in the user query.
    Note any differences between the user query requirements and the current plot. Based on the identified discrepancies, provide step-by-step instructions on how to modify the Python code to meet the user query requirements. Suggest improvements for better visualization practices, such as clarity, readability, and aesthetics, while ensuring the primary focus is on meeting the user's specified requirements.'''}
    ]
        messages[-1]["content"] += f"\n\n![plot](data:image/png;base64,{base64_image})"
        # print(messages[0]['content'])
        # Call the completion function to get feedback from GPT-4
        feedback = await self._call_openai_api(messages[0]['content'], messages[-1]['content'])
        # print(feedback)
        return feedback

    async def feedback_aggregation(self, code, feedback, data_description):
        code_with_feedback = ""
        code_with_feedback += f"------\nData Description:{data_description}\nInitial code:\n{code}\nFeedback :\n{feedback}"
        return code_with_feedback
    
    async def get_code_content_with_feedback(self, user_query, data_path_list, img_file_path):
        print(f'사용하는 모델 명 : {self.model}')
        initial_code, tmp_img_path, error_message, data_description = await self.get_code_content(user_query, data_path_list, img_save_path=img_file_path)
        if not error_message:
            visual_feedback = await self.visual_feedback(user_query, initial_code, img_save_path = tmp_img_path, data_description=data_description)
            code_with_feedback = await self.feedback_aggregation(initial_code, visual_feedback, data_description)
        else:
            code_with_feedback = f'''------\nData Description:{data_description}\n\nInitial Code: {initial_code}\n\nThere are some errors in the code you gave:
{error_message}
please correct the errors.
Then give the complete code and don't omit anything even though you have given it in the above code.'''
        # print('#'*5, 'Code With Feedback', '#' * 5)
        # print(code_with_feedback)
        prompt = f"""
## Original Request:
{user_query}

## Feedback for Improvement:
The following feedback has been generated to improve the code. Please consider these suggestions when modifying the code:
{code_with_feedback}

## Revised Code:
Modify the code based on the feedback above to fulfill the original request. Ensure the code is fully executable and ready to generate the plot. The code must work without errors and should include necessary imports and error handling. If there are nothing to revised than just export the Initial code above. The code should be outputted in the following format:
```python
# Your revised code here
plt.show()
```
Code Must end in plt.show() and don't save figure by plt.savefig()
"""
        try_count = 0
        while try_count < 4:
            print('Getting Code with Max4 trials')
            response_text = await self._call_openai_api(SYSTEM_PROMPT_CODE_GEN, prompt)
            # print(response_text)
            match = re.search(r"```python(.*?)```", response_text, flags=re.DOTALL)
            # print(match)
            if match:
                code = match.group(1).strip()
                code_n = code.replace("plt.show()", f"\nplt.close('all')")
                try:
                    exec(code_n)
                    log = True
                except:
                    log = False
                if log:
                    # print(code)
                    print('Stop Before Max4')
                    return code, prompt
                else:
                    print(f'Try Again {try_count+1}')
                    try_count += 1
            else:
                print(f'Try Again {try_count+1}')
                try_count += 1
        print('No code unitl Max4')
        return code, prompt
    
        # response_text = await self._call_openai_api(SYSTEM_PROMPT_CODE_GEN, prompt)
        # # print('#'*5, 'Final Code based on Feedback', '#' * 5)
        # # print(response_text)
        # match = re.search(r"```python(.*?)```", response_text, flags=re.DOTALL)
        # # match = re.search(r"```python(.*?)plt\.show\(\)", response_text, flags=re.DOTALL)
        # if match:
        #     return match.group(1).strip(), prompt
        #     # print(code)
        # else:
        #     return None, prompt