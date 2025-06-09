import asyncio
import re
import os
import pandas as pd
import base64
import io
from PIL import Image
import base64
from io import BytesIO
from openai import AsyncOpenAI
from dotenv import load_dotenv
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential, stop_after_delay,
)

# Load environment variables
load_dotenv()

class VisPath5:
    def __init__(self, api_key, base_url, model, temperature=0.2):
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("BASE_URL")
        self.model = model
        self.temperature = temperature
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        self.system_prompt_expansion='''According to the user query, expand and solidify the query into a step by step detailed instruction (or comment) on how to write python code to fulfill the user query's requirements. Import the appropriate libraries. Pinpoint the correct library functions to call and set each parameter in every function call accordingly.'''

        self.template_expansion ="""
            Generate five distinct extended queries based on the given original query.  
            Each extended query should be written in a Chain of Thought (CoT) style, explicitly outlining the reasoning and step-by-step methodology to achieve the goal.  

            ### **Key Requirements:**  
            - Do NOT change the goal or instructions given in the original query.  
            - Propose **five different methodologies** to achieve the goal while maintaining the original intent.  
            - Each extended query must be structured in a step-by-step CoT format, explaining **why** each step is necessary.  
            - The different methodologies should vary in terms of **data processing, visualization techniques, or computation strategies**.  
            - Ensure that the data description is analyzed and incorporated into the query design.  
            - If no data description is provided, then strictly follow the original query.  

            ### **Variations in Approach (Examples):**  
            - Using different **data processing techniques** (e.g., pandas, NumPy, direct iteration)  
            - Implementing various **visualization strategies** (e.g., different styles of plots)  
            - Exploring alternative **computation methods** (e.g., vectorized operations, grouped aggregations, iterative filtering)  

            ### **Input:**  
            The original query is: {ori_query}  

            Data description is: {data_description}  

            ### **Output Format:**  
            Return the output **ONLY** in the following Python list format:  
            ```[query_text_1, query_text_2, query_text_3, query_text_4...]```  
            """
        
        self.system_prompt_codegen="""You are an expert on data visualization code generation. You should think step by step, and write the generated code in the format of ```python...```, where ... indicates the generated code. Code Must end in plt.show() and don't save figure by plt.savefig() and don't save anything else either."""
        
        self.template_codegen="""\
            Based on the user's query and data description, generate Python code using Visualization Library like `matplotlib.pyplot` or 'seaborn' to create the requested plot. Ensure the code is outputted within the format ```...```, where ... indicates the generated code. Please make sure to generate the code according to the data description below. Do not include unnecessary code for saving plot figure, saving dataframe, or other unrelated tasks when generating the code. End with plt.show() and do not include anything after that.
            You Must use data_path provided in Data Description when loading data with pd.read_csv().

            User query: {query}

            Data description: {data_description}
            """

        self.system_prompt_aggregation="""
        You are an expert in analyzing, improving, and synthesizing data visualization code. 
        Your role is to evaluate multiple versions of visualization code based on user queries and data descriptions, 
        integrate feedback effectively, and generate a final version that best meets the user's requirements."
"""

        self.template_aggregation="""\
    Think step by step and plan before generating the final code.

You will be given:
- **User Query**: Instructions on how the user wants the data visualization (plot) to be performed.
- **Data Description**: Details about the dataset, including file paths and summaries.
- **Code for Aggregation with Corresponding Feedback**: five different versions of the data visualization code, each paired with its respective feedback highlighting mismatches with the user’s requirements and areas for improvement.

### **Your Task:**
1. **Understand the User's Intent**  
   - Analyze the **User Query** to extract key visualization requirements, constraints, and goals.
   - Carefully review the **Data Description** to ensure the final visualization correctly utilizes the given dataset.

2. **Evaluate the Provided Code Versions & Feedback**  
   - Examine all five versions of the code.
   - Review the feedback for each version and identify common issues, missing elements, and improvement points.
   - Determine which parts of each version align well with the user's requirements.

3. **Synthesize the Best Final Version**  
   - Construct a final version that effectively **integrates the best aspects** of the provided codes while addressing all necessary corrections from the feedback.  
   - Ensure the final code adheres to **high readability, clarity, and maintainability** while fully complying with the user’s original instructions.  
   - Eliminate unnecessary complexity while maintaining functionality.

4. **Output the Final Version**  
   - Provide the optimized final version inside a properly formatted code block:
     ```
     ```python
     # Final optimized code
     ...
     ```
     ```

### **Inputs:**
- **User Query:** {ori_query}
- **Data Description:** {data_description}
- **Code for Aggregation with Corresponding Feedback:** {code_for_aggregation}
"""

    def describe_data(self, data_path):
        # Read the data file using pandas
        data = pd.read_csv(data_path)
        description = {
            "data_path": data_path,
            "columns": list(data.columns),
            "dtypes": data.dtypes.to_dict(),
            "shape": data.shape,
            "sample": data.head(3).to_dict()
        }
        return str(description)

    def describe_data_list(self, data_path_list):
        try:
            data_description_list = []
            for data_path in data_path_list:
                root_path = os.path.abspath(os.path.dirname(os.curdir))
                data_path = os.path.normpath(os.path.join(root_path, data_path))
                single_data_description = self.describe_data(data_path)
                data_description_list.append(single_data_description)
            data_description = "[" + "], [".join(data_description_list) + "]"
        except:
            data_description = 'There is no dataset provided, please generate code based on the content of query.'
            
        return data_description

    # async def call_openai_api(self, system_prompt, user_content):
    #     while True:
    #         try:
    #             response = await self.client.chat.completions.create(
    #                 model=self.model,
    #                 messages=[
    #                     {"role": "system", "content": system_prompt},
    #                     {"role": "user", "content": user_content}
    #                 ],
    #                 temperature=self.temperature
    #             )
    #             return response.choices[0].message.content
    #         except Exception as e:
    #             print(f"API call failed with error: {e}. Retrying...")

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
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call failed with error: {e}. Retrying...")
            raise e

    async def expansion_agent(self, ori_query, data_description):
        print('Expansing agent')
        prompt = self.template_expansion.format(
                ori_query=ori_query,
                data_description=data_description
                )
        response_text = await self.call_openai_api(self.system_prompt_expansion, prompt)
        # match = re.search(r'\[.*?\]', response_text, flags=re.DOTALL)
        match = re.search(r'\[.*\]', response_text, flags=re.DOTALL)
        if match:
            generated_query = match.group().strip()
            try:
                parsed_list = ast.literal_eval(generated_query)
                # print(parsed_list)
                if isinstance(parsed_list, list) and parsed_list:
                    return parsed_list
                else:
                    return None
            except (ValueError, SyntaxError) as e:
                # print(e)
                return None
        else:
            print("No match found")
            return None            

    async def single_codegen(self, query, data_description):
        # Call API and return the generated code
        prompt=self.template_codegen.format(
                query=query,
                data_description=data_description,
                )
        print('Single Code Generating...')
        response_text = await self.call_openai_api(self.system_prompt_codegen, prompt)
        match = re.search(r"```python(.*?)```", response_text, flags=re.DOTALL)
        if match:
            code_content = match.group(1).strip()
            return code_content
        else:
            return None

    async def codegen_agent(self, ori_query, extend_query_list, data_description):
        print('Code Generating')
        try:
            tasks = [self.single_codegen(query, data_description) for query in extend_query_list]
            generated_code_list = await asyncio.gather(*tasks)

            # ori_generated_code = await self.single_codegen(ori_query, data_description)
            # generated_code_list.append(ori_generated_code)

            return generated_code_list
        except:
            return None
        
    # Function to execute the code and save the plot as an image
    def code_to_image(self, code, img_path):
        exec_globals = {"plt": plt}  # Define the global context for the code execution
        exec_locals = {}  # Local context for the code execution
        code = code.replace("plt.show()", f"plt.savefig('{img_path}')\nplt.close('all')")  # Replace plt.show() with savefig to save the image
        try:
            exec(code, exec_globals, exec_locals)  # Execute the code
            return True, 'No Error'  # Return True if the code executed successfully
        except Exception as e:
            return False, f'''There are some errors in the code you gave:
{str(e)}
please correct the errors.'''

    # Function to encode the image to base64 format
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def visual_feedback(self, ori_query, code, img_path):
        print("Visaul Feedback generating...")
        # Save the plot as an image using the provided code
        success, error_message = self.code_to_image(code, img_path)

        if not success:
            base64_image = error_message  # Return an error message if image saving fails
        else:
            # Encode the image to base64 for sending as a message
            base64_image = self.encode_image(img_path)
        # Prepare the messages for GPT-4o, including the system prompt, user prompt with code and query, and the image
        messages = [
            {
                "role": "system",
                "content": '''
                    Given a piece of code, a user query, and an image of the current plot, 
                    please determine whether the plot has faithfully followed the user query. 
                    
                    Your task is to provide instruction to make sure the plot has strictly 
                    completed the requirements of the query. Please output a detailed step by step 
                    instruction on how to use python code to enhance the plot. If the plot image 
                    is missing, check the error message that has occurred in the code. 
                    
                    Provide clear, essential instructions to modify the code based on the analysis, 
                    focusing only on the discrepancies between the plot and the user query. 
                    Avoid unnecessary feedback or suggestions. Do not output the final modified code, 
                    only the instructions.'''
            },
            {
                "role": "user",
                "content" : f'''
                    Here is the code: [Code]:
                        """
                        {code}
                        """

                        Here is the user query: [Query]:
                        """
                        {ori_query}
                        """
                    Carefully analyze the provided Python code, user query, and the plot image (if available) 
                    to evaluate if the generated plot meets the user query requirements. If the plot image is missing,
                    check the error message that has occurred in the code. 
                    
                    Compare the plot with the user query requirements, highlight discrepancies, and provide clear, 
                    essential instructions to modify the code accordingly.

                    Additionally, suggest improvements for better visualization, focusing on clarity, readability, 
                    and alignment with the user's objectives.
                    
                    Provide step-by-step instructions for code modification based on the analysis, focusing only on necessary 
                    corrections. Do not provide the final modified code, only the instructions for fixing the discrepancies.
                    '''
            }
        ]
        
        messages[-1]["content"] += f"\n\n![plot](data:image/png;base64,{base64_image})"
        feedback = await self.call_openai_api(messages[0]['content'], messages[-1]['content'])
        print('Feedback is done')
        return feedback


    async def feedback_aggregation(self, ori_query, img_path, generated_code_list):
        def add_idx(path, idx):
            parts = path.rsplit('.', 1)  
            if len(parts) == 2:
                return ".".join([f"{parts[0]}_path{idx}", parts[1]]) 
            return path  
        try:
            new_img_path_list = [add_idx(img_path, idx) for idx, _ in enumerate(generated_code_list)]
            tasks = [self.visual_feedback(ori_query, code, new_img_path) for code, new_img_path in zip(generated_code_list, new_img_path_list)]
            generated_feedback_list = await asyncio.gather(*tasks)

            code_with_feedback = ""
            for i in range(len(generated_code_list)):
                code_with_feedback += f"------\ncode{i+1}:\n{generated_code_list[i]}\nfeedback{i+1}:\n{generated_feedback_list[i]}\n"
            print("Feedback Aggregation is done")
            return code_with_feedback
        except:
            return None


    async def get_code_content(self, ori_query, data_path_list, img_file_path):
        print(f'Model name in use : {self.model}')
        data_description = self.describe_data_list(data_path_list)
        extend_query_list = await self.expansion_agent(ori_query, data_description)
        generated_code_list = await self.codegen_agent(ori_query, extend_query_list, data_description)
        code_with_feedback = await self.feedback_aggregation(ori_query, img_file_path, generated_code_list)

        prompt = self.template_aggregation.format(
                ori_query=ori_query,
                data_description=data_description,
                code_for_aggregation=code_with_feedback
                )

        response_text = await self.call_openai_api(self.system_prompt_codegen, prompt)
        match = re.search(r"```python(.*?)```", response_text, flags=re.DOTALL)
        if match:
            code = match.group(1).strip()
            return code, extend_query_list, generated_code_list, prompt
        else:
            return None, extend_query_list, generated_code_list, prompt