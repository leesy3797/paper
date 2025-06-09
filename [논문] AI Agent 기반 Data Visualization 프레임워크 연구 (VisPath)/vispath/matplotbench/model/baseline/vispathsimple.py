import asyncio
import re
import os
import pandas as pd
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

class VisPathSimple:
    def __init__(self, api_key=None, base_url=None, model="gpt-4o-mini", system_prompt_expansion=None, system_prompt_codegen=None, template_expansion=None, template_codegen=None, system_prompt_aggregation=None, template_aggregation=None, temperature=0.2):
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
            Generate three distinct extended queries based on the given original query.  
            Each extended query should be written in a Chain of Thought (CoT) style, explicitly outlining the reasoning and step-by-step methodology to achieve the goal.  

            ### **Key Requirements:**  
            - Do NOT change the goal or instructions given in the original query.  
            - Propose **three different methodologies** to achieve the goal while maintaining the original intent.  
            - Each extended query must be structured in a step-by-step CoT format, explaining **why** each step is necessary.  
            - The different methodologies should vary in terms of **data processing, visualization techniques, or computation strategies**.  
            - Ensure that the data description is analyzed and incorporated into the query design.  
            - If no data description is provided, then strictly follow the original query.  

            ### **Variations in Approach (Examples):**  
            - Using different **data processing techniques** (e.g., pandas, NumPy, direct iteration)  
            - Implementing various **visualization strategies** (e.g., different libraries, different styles of plots)  
            - Exploring alternative **computation methods** (e.g., vectorized operations, grouped aggregations, iterative filtering)  

            ### **Input:**  
            The original query is: {ori_query}  

            Data description is: {data_description}  

            ### **Output Format:**  
            Return the output **ONLY** in the following Python list format:  
            ```[query_text_1, query_text_2, query_text_3]```  
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
        and generate a final version that best meets the user's requirements."
        """


        self.template_aggregation="""\
        Think step by step and plan before generating the final code.

        You will be given:
        - **User Query**: Instructions on how the user wants the data visualization (plot) to be performed.
        - **Data Description**: Details about the dataset, including file paths and summaries.
        - **Different Versions of Codes**: Three different versions of the data visualization code

        ### **Your Task:**
        1. **Understand the User's Intent**  
        - Analyze the **User Query** to extract key visualization requirements, constraints, and goals.
        - Carefully review the **Data Description** to ensure the final visualization correctly utilizes the given dataset.

        2. **Evaluate the Provided Code Versions**  
        - Examine all three versions of the code in Different Versions of Codes below.
        - Determine which parts of each version align well with the user's requirements.

        3. **Synthesize the Best Final Version**  
        - Construct a final version that effectively meets the user's requirements perfectly.   
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
        - **Different Versions of Codes:** {code_for_aggregation}
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
        if extend_query_list:
            tasks = [self.single_codegen(query, data_description) for query in extend_query_list]
            generated_code_list = await asyncio.gather(*tasks)

            # ori_generated_code = await self.single_codegen(ori_query, data_description)
            # generated_code_list.append(ori_generated_code)
        else:
            generated_code_list = []
            ori_generated_code = await self.single_codegen(ori_query, data_description)
            generated_code_list.append(ori_generated_code)
        return generated_code_list

    async def get_code_content(self, ori_query, data_path_list):
        print(f'사용하는 모델 명 : {self.model}')
        data_description = self.describe_data_list(data_path_list)
        # print(data_description)
        extend_query_list = await self.expansion_agent(ori_query, data_description)
        # print(extend_query_list)
        generated_code_list = await self.codegen_agent(ori_query, extend_query_list, data_description)
        # print(generated_code_list)
        code_for_aggregation="{" + "}\n\n{".join(generated_code_list) + "}"        
        # print(code_for_aggregation)
        prompt = self.template_aggregation.format(
                ori_query=ori_query,
                data_description=data_description,
                code_for_aggregation=code_for_aggregation
                )
        response_text = await self.call_openai_api(self.system_prompt_aggregation, prompt)
        match = re.search(r"```python(.*?)```", response_text, flags=re.DOTALL)
        if match:
            return match.group(1).strip(), extend_query_list, generated_code_list
        else:
            return None, extend_query_list, generated_code_list