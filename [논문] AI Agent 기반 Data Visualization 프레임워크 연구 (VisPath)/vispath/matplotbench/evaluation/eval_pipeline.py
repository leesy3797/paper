import base64
import json
import sys
sys.path.insert(0, sys.path[0]+"/../")
from loguru import logger # type: ignore
import os
import re
import ast
import shutil
import random
import glob
import asyncio
import io
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import image_to_base64, code_to_image
from config.setting import OPENAI_API_KEY, OPENAI_BASE_URL, GEMINI_API_KEY, GEMINI_BASE_URL
from model.baseline import ZeroShot, CoT, Chat2Vis, MatplotAgent, VisPathSimple, VisPath, VisPath2, VisPath4, VisPath5, VisPath6, VisPath7, VisPath8, VisPathExecute
from eval_vis import gpt_4_evaluate, gpt_4v_evaluate
from openai import OpenAI
import gc

from dotenv import load_dotenv

load_dotenv()

def create_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Directory '{directory}' already exists. Contents removed.")
    
    os.makedirs(directory)
    print(f"Directory '{directory}' created successfully.")

def load_benchmark_data(benchmark, test_sample_id):
    print('Start Loading Benchmark Data')
    if benchmark == 'matplotbench':
        data = pd.read_csv('matplotbench\dataset\matplotbench_data.csv')
        id = data.loc[test_sample_id - 1, 'id']
        user_query = data.loc[test_sample_id - 1, 'simple_instruction'] # Matplotagent and ours use simple instruction for evaluation
        # user_query = data.loc[test_sample_id - 1, 'expert_instruction']
        ground_truth = data.loc[test_sample_id - 1, 'img_bs64']
        data_path = None if id <= 75 else [f'matplotbench/dataset/data/{id}/data.csv']
    return str(id), user_query, ground_truth, data_path if data_path else None

def extract_score(pattern, text):
    match = re.findall(pattern, text, re.DOTALL)
    return int(match[0]) if match else 0

def save_logs(log_path, logs):
    with open(log_path, 'w') as json_file:
        json.dump(logs, json_file, indent=4)

async def mainworkflow(test_sample_id, benchmark, baseline, model):
    if model == 'gpt-4o-mini':
        MODEL = 'gpt-4o-mini'
        API_KEY = os.environ['OPENAI_API_KEY'] # Insert your key in config > setting.py and change into OPENAI_API_KEY
        BASE_URL = OPENAI_BASE_URL
    elif model == 'gemini-2.0-flash':
        MODEL = 'models/gemini-2.0-flash'
        API_KEY = os.environ['GEMINI_API_KEY'] # Insert your key in config > setting.py and change into GEMINI_API_KEY
        BASE_URL = GEMINI_BASE_URL
    elif model == 'gpt-4o':
        MODEL = 'gpt-4o'
        API_KEY = os.environ['OPENAI_API_KEY'] # Insert your key in config > setting.py and change into OPENAI_API_KEY
        BASE_URL = OPENAI_BASE_URL
    
    BASELINE_MAPPING = {
        "zeroshot" : ZeroShot(
            api_key=API_KEY, 
            base_url=BASE_URL, 
            model=MODEL
        ),
        "cot" : CoT(
            api_key=API_KEY, 
            base_url=BASE_URL, 
            model=MODEL
        ),
        "chat2vis" : Chat2Vis(
            api_key=API_KEY, 
            base_url=BASE_URL, 
            model=MODEL   
        ),
        "matplotagent" : MatplotAgent(
            api_key=API_KEY, 
            base_url=BASE_URL, 
            model=MODEL
            ),
        "vispath" : VisPath(
            api_key=API_KEY, 
            base_url=BASE_URL, 
            model=MODEL
            ),
        "vispathsimple" : VisPathSimple(
            api_key=API_KEY, 
            base_url=BASE_URL, 
            model=MODEL
            ),
        "vispath2" : VisPath2(
            api_key=API_KEY, 
            base_url=BASE_URL, 
            model=MODEL
            ),
        "vispath4" : VisPath4(
            api_key=API_KEY, 
            base_url=BASE_URL, 
            model=MODEL
            ),
        "vispath5" : VisPath5(
            api_key=API_KEY, 
            base_url=BASE_URL, 
            model=MODEL
            ),
        "vispath6" : VisPath6(
            api_key=API_KEY, 
            base_url=BASE_URL, 
            model=MODEL
            ),
        "vispath7" : VisPath7(
            api_key=API_KEY, 
            base_url=BASE_URL, 
            model=MODEL
            ),
        "vispath8" : VisPath8(
            api_key=API_KEY, 
            base_url=BASE_URL, 
            model=MODEL
            ),
        "vispathexecute" : VisPathExecute(
            api_key=API_KEY, 
            base_url=BASE_URL, 
            model=MODEL
        )
    } # Add your own model  

    directory = f'{benchmark}/result/{baseline}/{model}/example_{test_sample_id}'
    create_directory(directory)
    id, user_query, ground_truth, data_path = load_benchmark_data(benchmark, test_sample_id)

    logger.info('=========Baseline Selection & Setting Image Save Path=========')
    model = BASELINE_MAPPING[baseline]
    img_save_path = f'{directory}/{id}.png'
    print('End Baseline Selection & Image Save Path....')

    logger.info('=========Generating Plot Code=========')
    print('End Loading Benchmark Data')
 
    print('Generating Code...')

    if baseline in ['zeroshot', 'cot', 'chat2vis']:
        code = await model.get_code_content(user_query, data_path_list=data_path)

    if baseline in ['matplotagent']:
        code, llms_feedback = await model.get_code_content_with_feedback(user_query, data_path_list=data_path, img_file_path=img_save_path)

    if baseline in ['vispathsimple']:
            code, extend_query_list, generated_code_list = await model.get_code_content(user_query, data_path_list=data_path)
        
    if baseline in ['vispath', 'vispath2', 'vispath4', 'vispath5', 'vispath6', 'vispath7', 'vispath8', 'vispathexecute']:
        code, extend_query_list, generated_code_list, llms_feedback = await model.get_code_content(user_query, data_path_list=data_path, img_file_path=img_save_path)

    print('End Generating Code!!!')

    logger.info('=========Execute Code & Save Final Image=========')
    code_to_image(code, img_save_path)
    executable = (True if os.path.exists(img_save_path) else False)
    print('End executing code & save image')

    logs = {
        "ID": id,
        "Executable Score": 1 if executable else 0,
        "User Query": user_query,
        "Final Code": code,
    }

    logger.info(f'Executability of {id} = {executable}')
    
    if executable:
        logger.info('=========Evaluating Code Score========')
        
        try:
            code_score = await gpt_4_evaluate(code, user_query, img_save_path)
            logs['Code Evaluation'] = code_score
            logs["Code Score"] = extract_score(r'.*?\[FINAL SCORE\]: (\d{1,3})', code_score)
            print('End Evaluate Code Score')
        except:
            logs['Code Evaluation'] = ""
            logs["Code Score"] = 0

        logger.info('=========Evaluating Plot Score========')
        
        try:
            plot_score = await gpt_4v_evaluate(ground_truth, image_to_base64(img_save_path))
            logs['Plot Evaluation'] = plot_score
            logs["Plot Score"] = extract_score(r'.*?\[FINAL SCORE\]: (\d{1,3})', plot_score)
            print('End Evaluate Plot Score')
        except:
            logs['Code Evaluation'] = ""
            logs["Plot Score"] = 0
    else:
        logs["Code Score"] = 0
        logs["Plot Score"] = 0

    if baseline in ['matplotagent_feedback']:
        logs["LLMs Feedback"] = llms_feedback

    if baseline in ['vispath', 'vispath2', 'vispath4', 'vispath5', 'vispath6', 'vispath7', 'vispath8', 'vispathexecute']:
        logs["Extended Query List"] = extend_query_list
        logs["Generated_code_list"] = generated_code_list
        logs["LLMs Feedback"] = llms_feedback

    logger.info('=========Saving Total Score in JSON========')
    log_path = f'{directory}/result_{benchmark}_{baseline}_{id}.json'
    save_logs(log_path, logs)
     
    gc.collect()
    del model