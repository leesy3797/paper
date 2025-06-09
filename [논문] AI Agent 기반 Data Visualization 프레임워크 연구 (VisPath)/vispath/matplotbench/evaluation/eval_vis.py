import os
import sys
import pandas as pd
from openai import AsyncOpenAI
from utils import image_to_base64, base64_to_image
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, sys.path[0]+"/../")

async def gpt_4_evaluate(code, query, image):
    print('GPT4o Evaluating...')
    executable = 'True' if os.path.exists(image) else 'False'
    API_KEY = os.getenv("API_KEY")
    client = AsyncOpenAI(
        api_key=API_KEY
    )
    while True:
        try:
          response = await client.chat.completions.create(
              model="gpt-4o-mini", # gpt-4o
              temperature=0.2,
              messages=[
                  {
                      "role": "user",
                      "content": [
                          {
                              "type": "text",
                              "text": f'''You are an excellent judge at evaluating generated code given an user query. You will be giving scores on how well a piece of code adheres to an user query by carefully reading each line of code and determine whether each line of code succeeds in carrying out the user query.
                              
                              A user query, a piece of code and an executability flag will be given to you. If the Executability is False, then the final score should be 0.

                              **User Query**: {query}

                              **Code**:
                              """
                              {code}
                              """
                              
                              **Executability**: {executable}
                              
                              Carefully read through each line of code. Scoring can be carried out in the following aspect:
                              1. Code correctness (Code executability): Can the code correctly achieve the requirements in the user query? You should carefully read each line of the code, think of the effect each line of code would achieve, and determine whether each line of code contributes to the successful implementation of requirements in the user query. If the Executability is False, then the final score should be 0.

                              After scoring from the above aspect, please give a final score. The final score is preceded by the [FINAL SCORE] token.
                              For example [FINAL SCORE]: 40. A final score must be generated.''',
                          },
                      ],
                  }
              ],
              max_tokens=1000,
          )
          return response.choices[0].message.content
        except Exception as e:
          print(f"API call failed with error: {e}. Retrying...")


async def gpt_4v_evaluate(ground_truth, generated_image_path):
    print('GPT4v Evaluating...')
    API_KEY = os.getenv("API_KEY")
    client = AsyncOpenAI(
        api_key=API_KEY
    )

    base64_image1 = ground_truth
    base64_image2 = generated_image_path
    while True:
      try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini", # gpt-4o
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f'''You are an excellent judge at evaluating visualization plots between a model generated plot and the ground truth. You will be giving scores on how well it matches the ground truth plot.
                            
                            The generated plot will be given to you as the first figure. If the first figure is blank, that means the code failed to generate a figure.
                            Another plot will be given to you as the second figure, which is the desired outcome of the user query, meaning it is the ground truth for you to reference.
                            Please compare the two figures head to head and rate them.
                            Suppose the second figure has a score of 100, rate the first figure on a scale from 0 to 100.
                            Scoring should be carried out in the following aspect:
                            1. Plot correctness: 
                            Compare closely between the generated plot and the ground truth, the more resemblance the generated plot has compared to the ground truth, the higher the score. The score should be proportionate to the resemblance between the two plots.
                            In some rare occurrence, see if the data points are generated randomly according to the query, if so, the generated plot may not perfectly match the ground truth, but it is correct nonetheless.
                            Only rate the first figure, the second figure is only for reference.
                            If the first figure is blank, that means the code failed to generate a figure. Give a score of 0 on the Plot correctness.
                            After scoring from the above aspect, please give a final score. The final score is preceded by the [FINAL SCORE] token.
                            For example [FINAL SCORE]: 40.''',
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image2}",
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image1}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )
        return response.choices[0].message.content
      except Exception as e:
          print(f"API call failed with error: {e}. Retrying...")