import base64
import logging
import os
import re

import torch
from config import get_model, get_react_parser
from utils.data_utils import load_jsonl, save_jsonl

torch.manual_seed(1234)

EVAL_VISUAL_PROMPT_ZH = """请判断图片是否与下面的[问题]一致，如果一致则回复“right”，不一致则回复“wrong”。
[问题]：{query}
"""

EVAL_VISUAL_PROMPT_EN = """Please judge whether the image is consistent with the [Question] below, if it is consistent then reply "right", if not then reply "wrong".
[Question]: {query}
"""

visualization_code_correctness = {
    'visualization-hard': None,
    'visualization-easy': None,
}


def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        a = base64.b64encode(image_file.read()).decode('utf-8')
    return a


async def judger_model_inference(judger_model_name, path, prompt=''):
    print('GPT4o-mini 평가 시작')
    output = ''
    # if judger_model_name == 'gpt-4o-mini':
    logging.warning('This is an example of `gpt-4o-mini`. '
                    'Please set the API key and use according to your actual situation.')
    from openai import OpenAI, AsyncOpenAI
    client = AsyncOpenAI(
        api_key = os.environ['OPENAI_API_KEY'],
        max_retries=10,
        timeout=30
        )
    content_list = []
    content_list.append({'type': 'text', 'text': prompt})
    input_images = []
    # for img in imgs:
        # print('이미지 확인 :', img)
        # if 'http' not in img:
    base64_image = encode_image(path)
    img = f'data:image/png;base64,{base64_image}'
    input_images.append(
        {
            'type': 'image_url', 
            'image_url': {'url' : img}
        }
    )
    content_list.extend(input_images)
    response = await client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{
            'role': 'user',
            'content': content_list,
        }],
        max_tokens=300,
    )
    output = response.choices[0].message.content
    logging.info(output)
    logging.info('=' * 60)
    return output


def extract_images(text):
    regex = re.compile(r'!\[fig-(.+)\]\((.+)\)')
    results = re.findall(regex, text)
    images = []
    for res in results:
        assert len(res) == 2
        if os.path.exists(res[1]):
            images.append(res[1])
    return images


def check_images_observation(text, images, model_name):
    start_flag = get_react_parser(model_name).observation
    for image in images:
        logging.info('Image'.center(60, '-'))
        logging.info(image)

        end_idx = text.find(image)
        tmp_text = text[:end_idx + len(image)]
        start_idx = tmp_text.rfind(start_flag)
        check_text = tmp_text[start_idx + len(start_flag):]

        logging.info('Observation'.center(60, '-'))
        logging.info(check_text)

        # As long as there exists correctly executed observation, we consider `True`
        if 'error:' not in check_text and 'Traceback' not in check_text:
            return True
    return False


eval_visual_prompt = {'zh': EVAL_VISUAL_PROMPT_ZH, 'en': EVAL_VISUAL_PROMPT_EN}


async def eval_visualization_acc(output_fname, model_name, judger_model_name='gpt-4o-mini', path_list=None):
    if judger_model_name == 'gpt-4o-mini':
        judger_model = None
    # elif judger_model_name in ['qwen-vl-chat', 'qwen-vl-plus']:
    #     if judger_model_name == 'qwen-vl-chat':
    #         logging.warning('In this benchmark of version 20231206, `Qwen-vl-chat` is no longer used as the '
    #                         'evaluation model for `Visualization` task.. If you insist on using it, '
    #                         'the evaluation results might differ from the official results.')
    #     judger_model = get_model(judger_model_name)
    # else:
    #     raise Exception('Not supported judger model.')

    one_action, one_action_right = 0, 0
    zero_action, zero_action_right = 0, 0
    print('Visualization 평가 진행중...')
    data_list = load_jsonl(output_fname)

    async def save_jsonl(data, path, progress=False, enabled=True):
        if not enabled:
            return
        with open(path, 'w', encoding='utf-8') as f:
            if progress:
                data = tqdm(data)
            for item in data:
                line = json.dumps(item, ensure_ascii=False)
                print(line, file=f)

        for item, path in zip(data_list, path_list):
            if 'visualization' not in item['tags']:
                continue
            # print('#' * 30)
            # print(item['query'])
            # print('#' * 30)
            line_id = item['idx']
            # path = f"C:/Users/LeeSeungYong/qwen/Qwen-Agent/benchmark/output_data/imgs_direct2vis/{line_id}.png"
            item['vis_acc'] = False
            if '<|im_end|>' in item['query']:
                one_action += 1
                prompt = item['query'].split('<|im_end|>')[0]
            else:
                zero_action += 1
                prompt = item['query']

            # print(line_id, path)
            if not os.path.exists(path):
                continue
            # images = extract_images(item['gen'])
            # if path:
            input_prompt = eval_visual_prompt[item.get('lang', 'en')]
            format_prompt = input_prompt.format(query=prompt)
            
            if not item['vis_acc']:
                output = await judger_model_inference(judger_model_name, path, format_prompt)
                # print('#' * 30)
                # print(item['query'], output, path)
                # print('#' * 30)
                if 'right' in output.lower():
                    item['vis_acc'] = True
                    if '<|im_end|>' in item['query']:
                        one_action_right += 1
                    else:
                        zero_action_right += 1
            else:    
                print(f'{item['idx']}번째 데이터는 Visual Evaluation 패스 ☑️')
                if item['vis_acc'] == True:
                    if '<|im_end|>' in item['query']:
                        one_action_right += 1
                    else:
                        zero_action_right += 1

    logging.info('*' * 60)
    logging.info('{:^60}'.format('Visualization Acc.'))
    logging.info('*' * 60)
    logging.info('Visualization-Hard count={}, Visualization-Hard right count={}, Visualization-Hard acc={:.2f}'.format(
        zero_action, zero_action_right, zero_action_right / zero_action * 100))
    logging.info('Visualization-Easy count={}, Visualization-Easy right count={}, Visualization-Easy acc={:.2f}'.format(
        one_action, one_action_right, one_action_right / one_action * 100))
    logging.info('all count={}, all right={}, all acc={:.2f}'.format(
        zero_action + one_action, zero_action_right + one_action_right,
        (zero_action_right + one_action_right) / (zero_action + one_action) * 100))

    visualization_code_correctness['visualization-hard'] = zero_action_right / zero_action * 100
    visualization_code_correctness['visualization-easy'] = one_action_right / one_action * 100

    error_data_list = [item for item in data_list if 'visualization' in item['tags'] and not item['vis_acc']]
    error_data_output_fname = os.path.splitext(output_fname)[0] + '_vis_error.jsonl'
    save_jsonl(error_data_list, error_data_output_fname)

    eval_done_list = [item for item in data_list if 'visualization' in item['tags'] and 'vis_acc' not in item]
    eval_done_list_fname = os.path.splitext(output_fname)[0] + '_eval_done_list.jsonl'
    save_jsonl(eval_done_list, eval_done_list_fname)

    return visualization_code_correctness


async def eval_visualization_acc(output_fname, model_name, judger_model_name='gpt-4o-mini', path_list=None):

    one_action, one_action_right = 0, 0
    zero_action, zero_action_right = 0, 0
    print('Visualization 평가 진행중...')
    data_list = load_jsonl(output_fname)

    # _eval_done_list.jsonl 파일 읽기
    eval_done_data_fname = os.path.splitext(output_fname)[0] + '_eval_done_list.jsonl'
    
    if os.path.exists(eval_done_data_fname):
        eval_done_data = load_jsonl(eval_done_data_fname)
        eval_done_ids = {item['idx'] : item['vis_acc'] for item in eval_done_data}  # 이미 평가된 데이터의 idx 집합 생성
        print(eval_done_ids)
    else:
        eval_done_ids = {}

    for item, path in zip(data_list, path_list):
        if 'visualization' not in item['tags']:
            continue
        
        if '<|im_end|>' in item['query']:
            one_action += 1
            prompt = item['query'].split('<|im_end|>')[0]
        else:
            zero_action += 1
            prompt = item['query']
            
        # 이미 평가된 항목 건너뛰기
        if item['idx'] in eval_done_ids:
            print(f'{item["idx"]}번째 데이터는 이미 Visual Evaluation이 완료되었습니다. 패스 ☑️')
            if eval_done_ids[item['idx']]:
                if '<|im_end|>' in item['query']:
                    one_action_right += 1
                else:
                    zero_action_right += 1
            item['vis_acc'] = eval_done_ids[item['idx']]
            continue

        item['vis_acc'] = False

        # 경로가 존재하면 진행
        if not os.path.exists(path):
            continue

        input_prompt = eval_visual_prompt[item.get('lang', 'en')]
        format_prompt = input_prompt.format(query=prompt)
        
        output = await judger_model_inference(judger_model_name, path, format_prompt)
        if 'right' in output.lower():
            item['vis_acc'] = True
            if '<|im_end|>' in item['query']:
                one_action_right += 1
            else:
                zero_action_right += 1

    # 결과 로그
    logging.info('*' * 60)
    logging.info('{:^60}'.format('Visualization Acc.'))
    logging.info('*' * 60)
    logging.info('Visualization-Hard count={}, Visualization-Hard right count={}, Visualization-Hard acc={:.2f}'.format(
        zero_action, zero_action_right, zero_action_right / zero_action * 100))
    logging.info('Visualization-Easy count={}, Visualization-Easy right count={}, Visualization-Easy acc={:.2f}'.format(
        one_action, one_action_right, one_action_right / one_action * 100))
    logging.info('all count={}, all right={}, all acc={:.2f}'.format(
        zero_action + one_action, zero_action_right + one_action_right,
        (zero_action_right + one_action_right) / (zero_action + one_action) * 100))

    visualization_code_correctness['visualization-hard'] = zero_action_right / zero_action * 100
    visualization_code_correctness['visualization-easy'] = one_action_right / one_action * 100

  
    def save_visualization_results(zero_action, zero_action_right, one_action, one_action_right):
        # 해당 로그만 저장할 파일 경로
        log_file = os.path.splitext(output_fname)[0] + '_vis_eval_result.txt'

        # 결과 로그 내용 구성
        result_log = "*" * 60 + "\n"
        result_log += "{:^60}".format('Visualization Acc.') + "\n"
        result_log += "*" * 60 + "\n"
        result_log += 'Visualization-Hard count={}, Visualization-Hard right count={}, Visualization-Hard acc={:.2f}\n'.format(
            zero_action, zero_action_right, zero_action_right / zero_action * 100)
        result_log += 'Visualization-Easy count={}, Visualization-Easy right count={}, Visualization-Easy acc={:.2f}\n'.format(
            one_action, one_action_right, one_action_right / one_action * 100)
        result_log += 'all count={}, all right={}, all acc={:.2f}\n'.format(
            zero_action + one_action, zero_action_right + one_action_right,
            (zero_action_right + one_action_right) / (zero_action + one_action) * 100)

        # 파일에 저장
        with open(log_file, 'a') as f:
            f.write(result_log)

    save_visualization_results(zero_action, zero_action_right, one_action, one_action_right)

    # error_data_list = [item for item in data_list if 'visualization' in item['tags'] and not item['vis_acc']]
    error_data_list = [item for item in data_list if 'visualization' in item['tags'] and not item.get('vis_acc', False)]
    # print(error_data_list)
    error_data_output_fname = os.path.splitext(output_fname)[0] + '_vis_error.jsonl'
    save_jsonl(error_data_list, error_data_output_fname)

    # eval_done_list에서 이미 vis_acc가 완료된 항목은 제외하도록 변경
    eval_done_list = [item for item in data_list if 'visualization' in item['tags'] and 'vis_acc' in item]
    # print(eval_done_list)
    eval_done_list_fname = os.path.splitext(output_fname)[0] + '_eval_done_list.jsonl'
    save_jsonl(eval_done_list, eval_done_list_fname)

    return visualization_code_correctness
