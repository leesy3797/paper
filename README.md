# Vispath: Automated Visualization Code Synthesis via Multi-Path Reasoning and Feedback-Driven Optimization

Vispath는 데이터 시각화를 위한 시각적 프로그래밍 에이전트입니다. 이 프로젝트는 MatplotBench과 QwenBench 두 가지 벤치마크를 통해 평가됩니다.

## 소개

Vispath는 자연어 명령어를 시각화 코드로 변환하는 자동화된 시스템입니다. 주요 특징은 다음과 같습니다:

- **다중 경로 추론**: 다양한 시각화 접근 방식을 탐색하고 최적의 해결책을 찾습니다.
- **피드백 기반 최적화**: 생성된 시각화의 품질을 평가하고 개선합니다.
- **코드 실행 및 검증**: 생성된 코드를 실행하고 결과를 검증합니다.

## 프로젝트 구조

```
vispath/
├── matplotbench/                # MatplotBench 벤치마크 관련 코드
│   ├── CV/                    # Fold 5 실험 결과
│   │   ├── cv1
│   │   ├── cv2                # cv2 디렉토리 밑에 정성평가용 데이터 위치
│   │   │   ├── [정성평가] 3개샘플데이터
│   │   │   └── [정성평가] 9개샘플데이터
│   │   ├── cv3
│   │   ├── cv4
│   │   └── cv5 
│   ├── dataset/               # 벤치마크 데이터셋(Matplotbench)
│   │   ├── data              # 76~100번 데이터
│   │   └── matplobench_data.csv  # Matplotbench 쿼리 (Simple Instruction, Expert Instruction)
│   ├── evaluation/           # 평가 관련 코드
│   │   ├── eval_batch.py     # 배치 평가 실행
│   │   ├── eval_pipeline.py  # 평가 파이프라인
│   │   ├── eval_vis.py       # 시각화 평가
│   │   └── utils.py          # 유틸리티 함수
│   ├── model/                # 실험 모델
│   │   └── baseline/         # 베이스라인 모델
│   │       ├── vispath.py    # Vispath
│   │       ├── vispath2.py   # Vispath v2
│   │       ├── vispath4.py   # Vispath v4
│   │       └── ...
│   ├── log/                  # 성공-실패 데이터 로그
│   │   └── vispath/gpt-4o-mini 
│   │       ├── completed_ids.txt  # evaluation 성공한 id값
│   │       └── failed_ids.txt     # evaluation 실패한 id값
│   │   └── vispath2/gpt-4o-mini 
│   │       ├── completed_ids.txt  # evaluation 성공한 id값
│   │       └── failed_ids.txt     # evaluation 실패한 id값
│   │   └── ...
│   ├── result/               # 성공-실패 데이터 로그
│   │   └── vispath/gpt-4o-mini 
│   │       ├── exaple_1      # 1번 케이스에 대한 로그 기록 (png, json)
│   │       ├── exaple_2      # 2번 케이스에 대한 로그 기록 (png, json)
│   │       └── ...
│   │   └── vispath2/gpt-4o-mini 
│   │       ├── exaple_1      # 1번 케이스에 대한 로그 기록 (png, json)
│   │       ├── exaple_2      # 2번 케이스에 대한 로그 기록 (png, json)
│   │       └── ...
│   │   └── ...
│   ├── baseline_result.json  # 최종 점수 파일
│   └── calculating_result.ipynb  # 최종 점수 계산 및 저장
│
├── qwenbench/                # QwenBench 벤치마크 관련 코드
│   ├── benchmark_code_interpreter_data/  # 벤치마크 데이터셋
│   │   ├── upload_file_clean            # raw data
│   │   └── eval_code_interpreter_v1.jsonl  # 중국어 쿼리 + 영어 쿼리
│   ├── eval_data/           # 추론 및 실행
│   │   ├── eval_code_interpreter_v1.jsonl  # 중국어 쿼리 + 영어 쿼리
│   │   └── eval_code_interpreter_v1.jsonl  # 영어 쿼리
│   ├── metrics/             # 평가 관련 코드
│   │   ├── code_execution.py
│   │   ├── gsm8k.py
│   │   └── visualization.py
│   ├── models/              # 실험 모델
│   │   ├── vispath.py      # Vispath
│   │   ├── vispath2.py     # Vispath v2
│   │   ├── vispath4.py     # Vispath v4
│   │   └── ...
│   ├── parser/             # (중요하지 않은 폴더)
│   │   └── ...
│   ├── prompt/             # (중요하지 않은 폴더)
│   │   └── ...
│   ├── utils/              # (중요하지 않은 폴더)
│   │   └── ...
│   ├── result/             # 실험 결과
│   │   └── vispath/gpt-4o-mini
│   │       ├── imgs_vispath    # 로그 파일 (png, json)
│   │       ├── logs_visapth_res_eval_done_list.jsonl   # 평가 완료 데이터
│   │       ├── logs_visapth_res_exec_error.jsonl       # 평가 미완료(에러) 데이터
│   │       ├── logs_visapth_res_vis_error.jsonl        # 코드 실행 실패 데이터
│   │       ├── logs_visapth_res_eval_done_list.jsonl   # 평가 결과
│   │       └── logs_vispath_res_vis_eval_result.txt    # 평가 점수 기록
│   ├── inference_and_execute_batch.py  # 추론 및 실행
│   ├── code_interpreter.py  # 코드 인터프리터
│   └── config.py           # 설정 파일
│
├── run.txt                # CLI 명령 프롬프트
└── requirements.txt       # 의존성 패키지
```

## 설치 방법

1. 저장소 클론:
```bash
git clone https://github.com/yourusername/vispath.git
cd vispath
```

2. 가상환경 생성 및 활성화:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. 의존성 설치:
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정:
`.env` 파일을 생성하고 다음 변수들을 설정합니다:
```
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=your_openai_base_url (건들 필요 없음)
GEMINI_API_KEY=your_gemini_api_key
GEMINI_BASE_URL=your_gemini_base_url (건들 필요 없음)
```

## 벤치마크 실행 방법

### MatplotBench 벤치마크

MatplotBench 벤치마크는 다양한 시각화 모델의 성능을 평가합니다. 
run.txt 파일 내에 있는 Cli 프롬프트 확인해서 필요한 케이스에 대해서 실행 

```bash
# GPT-4o-mini 모델로 평가
python matplotbench/evaluation/eval_batch.py --benchmark=matplotbench --baseline=vispath --model=gpt-4o-mini

# Gemini-2.0-flash 모델로 평가
python matplotbench/evaluation/eval_batch.py --benchmark=matplotbench --baseline=vispath --model=gemini-2.0-flash
```

사용 가능한 베이스라인:
- zeroshot
- cot
- chat2vis
- matplotagent
- vispath
- vispath2
- vispath4
- vispath5
- vispath6
- vispath7
- vispath8

### QwenBench 벤치마크

QwenBench 벤치마크는 시각화 작업에 대한 모델의 성능을 평가합니다.

```bash
# GPT-4o-mini 모델로 평가
python qwenbench/inference_and_execute_batch.py --task=visualization --model=gpt-4o-mini --baseline=vispath

# Gemini-2.0-flash 모델로 평가
python qwenbench/inference_and_execute_batch.py --task=visualization --model=gemini-2.0-flash --baseline=vispath
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 인용

이 프로젝트를 사용하실 때는 다음 형식으로 인용해 주세요:

```bibtex
@misc{seo2025automatedvisualizationcodesynthesis,
      title={Automated Visualization Code Synthesis via Multi-Path Reasoning and Feedback-Driven Optimization}, 
      author={Wonduk Seo and Seungyong Lee and Daye Kang and Hyunjin An and Zonghao Yuan and Seunghyun Lee},
      year={2025},
      eprint={2502.11140},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2502.11140}, 
}
``` 