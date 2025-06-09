import asyncio
import aiofiles
import argparse
import json
import logging
import os
from parser import ReActParser
from openai import AsyncOpenAI
import prettytable
import tqdm
from code_interpreter import code_interpreter
from config import get_model, get_react_parser, get_react_prompt, model_path_map
from datasets import load_dataset
from metrics.code_execution import eval_code_execution_rate
from metrics.gsm8k import eval_gsm8k_acc, is_correct
from metrics.visualization import eval_visualization_acc
from utils.code_utils import replace_upload_fname
from utils.data_utils import load_jsonl

from dotenv import load_dotenv
from models import VisPath, VisPath2, VisPath4, VisPath5, VisPath6, VisPath7, VisPath8
import shutil
# import chardet
import platform

# logging.basicConfig(level=logging.DEBUG)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)

# WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/workspace')
# os.makedirs(WORK_DIR, exist_ok=True)

if platform.system() == "Windows":
    pass
    # shutil.copytree("vispath/benchmark_code_interpreter_data/upload_file_clean", WORK_DIR + "vispath/qwenbench/upload_file", dirs_exist_ok=True)
    shutil.copytree("qwenbench/benchmark_code_interpreter_data/upload_file_clean", "qwenbench/upload_file", dirs_exist_ok=True)
else:
    # os.system("cp -r upload_file_clean {}/upload_file".format(WORK_DIR))
    os.system("cp -r upload_file_clean qwenbench/upload_file")

load_dotenv()

global_eval_result = {
    'code_executability': {
        'math': None,
        'visualization': None,
        'general': None,
    },
    'code_correctness': {
        'math': None,
        'visualization-hard': None,
        'visualization-easy': None,
    }
}


async def llm_with_plugin(args, query, item=None, exec_limit=3):
    exec_count = 0
    baseline = args.llm
    # Build ReAct prompt
    base_dir = os.path.join(os.getcwd(), 'qwenbench/upload_file')
    print(base_dir)
    
    upload_fname_list = [
        os.path.join(base_dir, upload_fname)
        for upload_fname in item['input_file_path']
    ] if item and 'input_file_path' in item else []
    
    lang = item['lang'] if item and 'lang' in item else 'en'
    print(upload_fname_list, lang)


    path = str(args.dir).replace('\\', '/') + f'/{item['idx']}.png'
    print(path)
    text = ''
    while exec_count < exec_limit:
        code = await baseline.generate(nl_query=query, tables=upload_fname_list, config = path)
        if code:
            text = code
            break
        else:
            exec_count += 1
            break
    return text

async def async_llm_with_plugin(args, query, item=None, exec_limit=3):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, llm_with_plugin, args, query, item, exec_limit)


def text_completion(llm, input_text, stop_words=[]):
    logging.info('Generating'.center(60, '='))
    logging.info('Input'.center(60, '-'))
    logging.info(input_text)

    output = llm.generate(input_text, stop_words)

    logging.info('Output'.center(60, '-'))
    logging.info(output)
    return output


async def async_call_tool(plugin_name, plugin_args_list, clear=False):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, call_tool, plugin_name, plugin_args_list, clear)

async def async_process_code_interpreter(item, writer):
    query = item['query']
    exec_limit = 3 if 'visualization' in item['tags'] else 1

    response = await llm_with_plugin(args=args, query=query, item=item, exec_limit=exec_limit)
    item['gen'] = response
    
    async with aiofiles.open(writer, 'a', encoding='utf-8') as f:
        await f.write(json.dumps(item, ensure_ascii=False) + '\n')

async def async_process_gsm8k(doc, writer):
    context = doc['question']
    completion = await async_llm_with_plugin(args=args, query=context)
    acc = is_correct(completion, doc['answer'])
    doc['completion'] = completion
    doc['acc'] = acc

    async with aiofiles.open(writer, 'a', encoding='utf-8') as f:
        await f.write(json.dumps(doc, ensure_ascii=False) + '\n')


async def async_sequential_processing(args, data_list, process_func, output_fname, batch_size=10):
    idx_list = []
    if os.path.exists(output_fname):
        with open(output_fname, 'r', encoding = 'utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'gen' not in data:
                    idx_list.append(data['idx'])

    to_process_list = [item for item in data_list if item['idx'] not in idx_list]
    logging.info(f'Number of data to be processed, excluding the executed ones: {len(to_process_list)} items ✅')
    
    for i in range(0, len(to_process_list), batch_size):
        batch = to_process_list[i:i + batch_size] 
        tasks = [process_func(item, output_fname) for item in batch]
        await asyncio.gather(*tasks)


def gather_eval_result(model_name):
    for metric in global_eval_result:
        logging.info(metric)
        table = prettytable.PrettyTable()
        table.field_names = ['model'] + list(global_eval_result[metric].keys())
        row_data = [model_name]
        for item in global_eval_result[metric].values():
            item = str(item) if not item else str(round(item, 2))
            row_data.append(item)
        table.add_row(row_data)
        logging.info('\n' + str(table))


async def eval_metrics(args, test_set, full_output_fname):
    # metrics
    assert os.path.exists(full_output_fname), f'Not Found File {full_output_fname}.'
    inference_res = load_jsonl(full_output_fname)
    assert len(inference_res) == len(test_set), f'There are still {len(test_set)-len(inference_res)} cases left.'

    # abs_output_fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), full_output_fname)
    abs_output_fname = full_output_fname

    if args.task == 'gsm8k':
        math_code_correctness = eval_gsm8k_acc(abs_output_fname)
        global_eval_result['code_correctness'].update(math_code_correctness)
    else:
        print("ABS_Output Fname: ", abs_output_fname)
        code_executability, path_list = eval_code_execution_rate(abs_output_fname, args, args.task, args.model)
        global_eval_result['code_executability'].update(code_executability)
        if args.task in ['all_ci', 'visualization'] and not args.eval_code_exec_only:
            visualization_code_correctness = await eval_visualization_acc(abs_output_fname, args.model, args.vis_judger, path_list)
            global_eval_result['code_correctness'].update(visualization_code_correctness)



async def async_main(args):
    args.output_path = os.path.join(os.path.abspath(os.getcwd()), f'qwenbench/result/{args.baseline}/{args.model}')
    args.output_fname = os.path.join(args.output_path, f'logs_{args.baseline}_res.jsonl')
    # print(args.output_path, args.output_fname)

    os.makedirs(args.output_path, exist_ok=True)
    full_output_fname = os.path.join(args.output_path, (args.output_fname or f'{args.task}_{args.model}_res.jsonl'))

    if not os.path.exists(full_output_fname):
        async with aiofiles.open(full_output_fname, 'w', encoding = 'utf-8'):
            logging.info(f'Create file {full_output_fname} done.')

    dir = args.output_path + f'/imgs_{args.baseline}'

    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    args.dir = dir

    # if args.task == 'gsm8k':
    #     dataset = load_dataset('gsm8k', 'main')
    #     test_set = dataset['test']
    # else:

    args.input_path = os.path.join(os.path.abspath(os.getcwd()), f'qwenbench/eval_data')
    eval_data_path = os.path.join(args.input_path, args.input_fname)
    test_set = [item for item in load_jsonl(eval_data_path) if args.task in item['tags']]

    if args.eval_only:
        await eval_metrics(args, test_set, full_output_fname)
    else:
        key = 'question' if args.task == 'gsm8k' else 'query'
        cache_question = [item[key] for item in load_jsonl(full_output_fname)] if not args.force else []
        data_list = [item for item in test_set if item[key] not in cache_question]

        process_func = {
            'gsm8k': async_process_gsm8k,
            'visualization': async_process_code_interpreter
        }.get(args.task, async_process_code_interpreter)

        await async_sequential_processing(args, data_list, process_func, full_output_fname, batch_size=args.batchsize)

        if not args.gen_exec_only:
            await eval_metrics(args, test_set, full_output_fname)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--task', type=str, default='all', choices=['all', 'gsm8k', 'visualization', 'general'])
    parser.add_argument('--output-path', type=str, default='output_data')
    parser.add_argument('--input-path', type=str, default='eval_data')
    parser.add_argument('-o', '--output-fname', type=str, default='')
    parser.add_argument('-i', '--input-fname', type=str, default='eval_code_interpreter_v3.jsonl') # If it's hard to read Chinese, then use eval_code_interpreter_v2.jsonl (Only English) 
    parser.add_argument('-f', '--force', action='store_true', default=False)
    parser.add_argument('--eval-only', action='store_true', default=False)
    parser.add_argument('--eval-code-exec-only', action='store_true', default=False)
    parser.add_argument('--gen-exec-only', action='store_true', default=False)
    parser.add_argument('--gen-only', action='store_true', default=False)
    parser.add_argument('--vis-judger',
                        type=str,
                        default="'gpt-4o-mini'")
                        # choices=['gpt-4-vision-preview', 'qwen-vl-chat', 'qwen-vl-plus'])
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    if args.model == 'gpt-4o-mini':
        MODEL = 'gpt-4o-mini'
        API_KEY = os.environ['OPENAI_API_KEY'] # Insert your key in config > setting.py and change into OPENAI_API_KEY
        BASE_URL = os.environ['OPENAI_BASE_URL']

    elif args.model == 'gemini-2.0-flash':
        MODEL = 'models/gemini-2.0-flash'
        API_KEY = os.environ['GEMINI_API_KEY'] # Insert your key in config > setting.py and change into GEMINI_API_KEY
        BASE_URL = os.environ['GEMINI_BASE_URL']

    llm = AsyncOpenAI(
        api_key = API_KEY,
        base_url = BASE_URL,
        max_retries=10,
        timeout=30
    )
    baselines = {
        'vispath' : VisPath(client = llm, model = args.model),
        'vispath2' : VisPath2(client = llm, model = args.model),
        'vispath4' : VisPath4(client = llm, model = args.model),
        'vispath5' : VisPath5(client = llm, model = args.model),
        'vispath6' : VisPath6(client = llm, model = args.model),
        'vispath7' : VisPath7(client = llm, model = args.model),
        'vispath8' : VisPath8(client = llm, model = args.model)
        }
    
    if not args.eval_only:
        args.llm = baselines[args.baseline]
        args.batchsize = 20
        logging.info(f'Init {args.baseline} done.')

    asyncio.run(async_main(args))
    # asyncio.run(async_main(args), debug=True)