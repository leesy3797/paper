import re
import asyncio
import pandas as pd
import os
import ast
import gc
from pathlib import Path
# from viseval import Dataset
# from viseval.agent import Agent, ChartExecutionResult
# import cairosvg
import base64
from .utils import show_svg
import aiofiles
import json
# from utils import show_svg
from dotenv import load_dotenv

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s') # Debugging mode

load_dotenv()

class VisPath7():
    def __init__(self, client, model='gpt-4o-mini', temperature = 0.2):
        self.client = client
        self.model = model
        self.temperature = temperature

        self.system_prompt_expansion='''According to the user query, expand and solidify the query into a step by step detailed instruction (or comment) on how to write python code to fulfill the user query's requirements. Import the appropriate libraries. Pinpoint the correct library functions to call and set each parameter in every function call accordingly.'''

        self.template_expansion ="""
            Generate seven distinct extended queries based on the given original query.  
            Each extended query should be written in a Chain of Thought (CoT) style, explicitly outlining the reasoning and step-by-step methodology to achieve the goal.  

            ### **Key Requirements:**  
            - Do NOT change the goal or instructions given in the original query.  
            - Propose **seven different methodologies** to achieve the goal while maintaining the original intent.  
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
            ```[query_text_1, query_text_2, query_text_3, query_text_4, query_text_5, query_text_6, query_text_7]``` 
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
- **Code for Aggregation with Corresponding Feedback**: seven different versions of the data visualization code, each paired with its respective feedback highlighting mismatches with the user’s requirements and areas for improvement.

### **Your Task:**
1. **Understand the User's Intent**  
   - Analyze the **User Query** to extract key visualization requirements, constraints, and goals.
   - Carefully review the **Data Description** to ensure the final visualization correctly utilizes the given dataset.

2. **Evaluate the Provided Code Versions & Feedback**  
   - Examine all seven versions of the code.
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

    # Function to execute the code and save the plot as an image
    def code_to_image(self, code, img_save_path):
        import matplotlib.pyplot as plt
        exec_globals = {"plt": plt}
        exec_locals = {}
        print('Start Executing Code and Save Image')
        try:
            code_n = code.replace("plt.show()", f"plt.savefig('{img_save_path}')\nplt.close('all')")
            exec(code_n, exec_globals, exec_locals)
            message = "Save Image Successfully!"
            print(message)
            return code, True, None
        except Exception as e:
            message = f'''There are some errors in the code you gave:
                {str(e)}
                please correct the errors.'''
            print(message)
            return code, False, message
        
    # Function to encode the image to base64 format
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def describe_data(self, data_path):
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
    
    def describe_data_list(self, data_path_list):
        try:
            data_description_list = []
            for data_path in data_path_list:
                print(data_path)
                single_data_description = self.describe_data(data_path)
                data_description_list.append(single_data_description)
            data_description = "[" + "], [".join(data_description_list) + "]"
        except:
            data_description = 'There is no dataset provided, please generate code based on the content of query.'

        return data_description
    
    async def expansion_agent(self, ori_query, data_description):
        print('Expansing Start')
        prompt = self.template_expansion.format(
                ori_query=ori_query,
                data_description = data_description
                )
        response_text = await self.call_openai_api(self.system_prompt_expansion, prompt)
        print('Expansing End')
        match = re.search(r'\[.*\]', response_text, flags=re.DOTALL)
        if match:
            generated_query = match.group().strip()
            try:
                parsed_list = ast.literal_eval(generated_query)
                if isinstance(parsed_list, list) and parsed_list:
                    return parsed_list
                else:
                    return None
            except (ValueError, SyntaxError) as e:
                return None
        else:
            print("No match found")
            return None        
        
    async def single_codegen(self, query, data_description, img_save_path):
        # Call API and return the generated code
        prompt=self.template_codegen.format(
                query=query,
                data_description=data_description,
                )
        print('Single Code Generating...')
        response_text = await self.call_openai_api(self.system_prompt_codegen, prompt)
        match = re.search(r"```python(.*?)```", response_text, flags=re.DOTALL)
        if match:
            code = match.group(1).strip()
            if 'plt.show()' not in code:
                code += '\nplt.show()'
            # img_save_path = os.path.join(os.path.dirname(os.getcwd(), '../..'), img_save_path)
            code, log, error_message = self.code_to_image(code,img_save_path)
            tmp_img_path = f'{img_save_path}'
            print('Code Successfully Generated')
            return code, None, tmp_img_path
        else:
            return code, error_message, tmp_img_path
        
    async def codegen_agent(self, ori_query, extend_query_list, data_description, img_file_path):
        def add_idx(path, idx):
            # path = os.path.join(os.path.dirname(os.getcwd(), '../..'), path)
            print(f"원래 경로 : {path}")
            parts = str(path).rsplit('.', 1)            
            new_path = ".".join([f"{parts[0]}_path{idx}", parts[1]])
            print(f"새로운 경로 : {new_path}")
            return str(new_path)
        try:
            print('PNG 저장 경로 확장')
            new_img_path_list = [add_idx(img_file_path, idx) for idx, _ in enumerate(extend_query_list)]
            print(new_img_path_list)
            print('Mulit Path Code Generating')
            tasks = [self.single_codegen(query, data_description, img_path) for query, img_path in zip(extend_query_list, new_img_path_list)]
            generated_code_list = await asyncio.gather(*tasks)
            # ori_generated_code = await self.single_codegen(ori_query, data_description)
            # generated_code_list.append(ori_generated_code)
            print('Multi Path Code Generated Finish')
            # print(generated_code_list[0])
            return generated_code_list
        except:
            return None
 

    async def visual_feedback(self, ori_query, code, img_save_path, error_msg):
        print('Starting Feedback...')
        # Encode the image to base64 for sending as a message
        if error_msg:
            base64_image = error_msg
            print('Return Error Message Instead Base64')
        else:
            try:
                base64_image = self.encode_image(img_save_path)
                print('Successfully Convert PNG to Base64')
            except:
                base64_image = "No Image"
                print('Fail to Encoding Base64')
        
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
        # print(messages[0]['content'])
        # Call the completion function to get feedback from GPT-4
        feedback = await self.call_openai_api(messages[0]['content'], messages[-1]['content'])
        # print(feedback)
        return feedback
    
    async def feedback_aggregation(self, ori_query, generated_code_list):
        try:
            print('각 Path 별 Visual Feedback 시작')
            tasks = [self.visual_feedback(ori_query, code, new_img_path, error_msg) for code, error_msg, new_img_path in generated_code_list]
            generated_feedback_list = await asyncio.gather(*tasks)
            print('Feedback Generated Successfully')
            code_with_feedback = ""
            for i in range(len(generated_code_list)):
                code_with_feedback += f"------\ncode{i+1}:\n{generated_code_list[i][0]}\nfeedback{i+1}:\n{generated_feedback_list[i]}\n"
            print("Feedback Aggregation is done")
            return code_with_feedback
        except:
            return None
    
        
    async def call_openai_api(self, system_prompt, user_content, tempertature = 0.2):
        # print('API 호출 중')
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=tempertature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call failed with error: {e}. Retrying...")
        
     
    async def generate(self, nl_query: str, tables: list[str], config = None):
        # img_file_path = config['img_path']
        
        data_description = self.describe_data_list(tables)
        # print(data_description)
        extend_query_list = await self.expansion_agent(nl_query, data_description)
        # print(extend_query_list)
        gc.collect()
        generated_code_list = await self.codegen_agent(nl_query, extend_query_list, data_description, config)
        print(generated_code_list)
        gc.collect()
        code_with_feedback = await self.feedback_aggregation(nl_query, generated_code_list)
        # print(code_with_feedback)
        
        print('Feedback 생성까지 완료')
        prompt = self.template_aggregation.format(
                ori_query=nl_query,
                data_description=data_description,
                code_for_aggregation=code_with_feedback
                )

        # parent_dir = config.parent  # logs/logs_vispath_feedback/37
        json_save_path = config
        # json_save_path = f'C:/Users/LeeSeungYong/qwen/Qwen-Agent/benchmark/{config}'
        json_save_path = json_save_path.split('.')[0] + '.json'
        print(json_save_path)
        
        response_text = await self.call_openai_api(self.system_prompt_codegen, prompt)
        # print('#'*5, 'Final Code based on Feedback', '#' * 5)
        # print(response_text)
        try:
            match = re.search(r"```python(.*?)```", response_text, flags=re.DOTALL)
            # match = re.search(r"```python(.*?)plt\.show\(\)", response_text, flags=re.DOTALL)
            if match:
                code = match.group(1).strip()
                save_data = {
                    "User Query" : nl_query,
                    "Extended Queries List" : extend_query_list,
                    "Generated Codes List" : [c[0] for c in generated_code_list],
                    "Generated Codes with Feedback": code_with_feedback,
                    "Final Code" : code,
                }

                # with open(str(json_save_path), 'w', encoding='utf-8') as f:
                #     json.dump(save_data, f, indent=4)
                async with aiofiles.open(str(json_save_path), 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(save_data, indent=4, ensure_ascii=False))
                gc.collect()
                print('Json 저장 완료')
                return code
                # print(code)
            else:
                save_data = {
                    "User Query" : nl_query,
                    "Extended Queries List" : extend_query_list,
                    "Generated Codes List" : [c[0] for c in generated_code_list],
                    "Generated Codes with Feedback": code_with_feedback,
                    "Final Code" : None,
                }

                # with open(str(json_save_path), 'w', encoding='utf-8') as f:
                #     json.dump(save_data, f, indent=4)
                async with aiofiles.open(str(json_save_path), 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(save_data, indent=4, ensure_ascii=False))
                gc.collect()
                print('Json 저장 완료')
                return None
        except:
            save_data = {
                "User Query" : nl_query,
                "Extended Queries List" : extend_query_list,
                "Generated Codes List" : [c[0] for c in generated_code_list],
                "Generated Codes with Feedback": code_with_feedback,
                "Final Code" : None,
            }

            # with open(str(json_save_path), 'w', encoding='utf-8') as f:
            #     json.dump(save_data, f, indent=4)
            async with aiofiles.open(str(json_save_path), 'w', encoding='utf-8') as f:
                await f.write(json.dumps(save_data, indent=4, ensure_ascii=False))
            gc.collect()
            print('Json 저장 완료')
            return None