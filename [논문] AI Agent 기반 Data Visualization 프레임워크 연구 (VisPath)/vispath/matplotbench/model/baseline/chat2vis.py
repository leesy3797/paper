import asyncio
import re
import os
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
import warnings

# Load environment variables
load_dotenv()

class Chat2Vis:
    def __init__(self, api_key=None, base_url=None, model="gpt-4o-mini", temperature=0.2):
        self.system_prompt = "You are a helpful assistant good at data visualization. The code you generate should be in the Markdown format like ```python...```."
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("BASE_URL")

        self.client = AsyncOpenAI(
            api_key = self.api_key,
            base_url = self.base_url
        )
        self.model = model
        self.temperature = temperature
        self.template_data = """\
        Data path:{data_path}

        Columns of {data_path}: {columns} .

        {columns_description}
        """

        self.template_all = """\
        Name the data provided below as df1, df2, ...

        Data description:

        {data_description}
        
        Instruction:

        Label the x and y axes appropriately. Add a title. Set the fig suptitle as empty. 

        Using Python version 3.9.12, create a script using those dataframes to graph the following:

        {nl_query}

        Below is the head of the code, you should write the code with them as the head, in the Markdwon format of ```python ...```.

        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        """
    
    async def call_openai_api(self, system_prompt, user_content):
        while True:
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

    def table_format(self, data: pd.DataFrame):
        # table format
        descriptions = []
        MAXIMUM_SAMPLES = 5  # Adjust if needed
        for column in data.columns:
            dtype = data[column].dtype
            description = None
            if dtype in [int, float, complex]:
                description = f"The column '{column}' is type {dtype} and contains numeric values."
            elif dtype == bool:
                description = f"The column '{column}' is type {dtype} and contains boolean values."
            elif dtype == object:
                # Check if the string column can be cast to a valid datetime
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(data[column], errors="raise")
                        dtype = "date"
                except ValueError:
                    # Check if the string column has a limited number of values
                    if data[column].nunique() / len(data[column]) < 0.5:
                        dtype = "category"
                    else:
                        dtype = "string"
            elif pd.api.types.is_categorical_dtype(data[column]):
                dtype = "category"
            elif pd.api.types.is_datetime64_any_dtype(data[column]):
                dtype = "date"

            if dtype == "date" or dtype == "category":
                non_null_values = data[column][data[column].notnull()].unique()
                n_samples = min(MAXIMUM_SAMPLES, len(non_null_values))
                samples = ( 
                    pd.Series(non_null_values)
                    .sample(n_samples, random_state=42)
                    .tolist()
                )
                values = "'" + "', '".join(samples) + "'"
                description = f"The column '{column}' has {dtype} values {values}"

                if n_samples < len(non_null_values):
                    description += " etc."
                else:
                    description += "."
            elif description is None:
                description = f"The column '{column}' is {dtype} type."

            descriptions.append(description)

        return " ".join(descriptions)

    async def get_code_content(self, user_query, data_path_list=None, ):
        # print(f'사용하는 모델 명 : {self.model}')
        data_description = ''
        if data_path_list:
            data_description_list = []
            for data_path in data_path_list:
                root_path = os.path.abspath(os.path.dirname(os.curdir))
                data_path = os.path.normpath(os.path.join(root_path, data_path))
                data = pd.read_csv(data_path)
                columns = "'" + "', '".join(list(data.columns)) + "'"
                columns_description = self.table_format(data)
                single_description = self.template_data.format(
                        data_path = data_path,
                        columns="'" + "', '".join(list(data.columns)) + "'", 
                        columns_description=columns_description,
                        )
                data_description_list.append(single_description)
            data_description = "[" + "], [".join(data_description_list) + "]"
        
        prompt = self.template_all.format(
            data_description=data_description,
            nl_query=user_query,
        )
        # print(prompt)

        response_text = await self.call_openai_api(self.system_prompt, prompt)
        match = re.search(r'```python\n(.*?)```', response_text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return None
