This is the official github repository for the paper: 🐼 [PANDA: Persona Attributes Navigation for Detecting and Alleviating Overuse Problem in Large Language Models](https://openreview.net/forum?id=MhxzXcjTka) (Accepted at EMNLP 2024 Main).

*Jinsung Kim(\*), Seonmin Koo(\*), and Heuiseok Lim*</br>
🏫 [NLP & AI Lab.](https://blpkorea.cafe24.com/wp/), Korea University, South Korea

## 🛠️ Installation
```bash
$ git clone https://github.com/jin62304/PANDA.git
```
```bash
# python_requires >=3.9
$ cd ./PANDA
$ pip install -r requirements.txt 
```
## 🚀 Usage
### Sample Data
- We provide a sample of data where dialogue topic information is tagged by persona attribute through human annotation.
- Each persona attribute has one or more topics.
```bash
## data path
$ ./data/panda_sample.json

## dialogue topic labels
topcis = {
  "0": "preferences:hobby/habit",
  "1": "preferences:food",
  "2": "preferences:others",
  "3": "occupations",
  "4": "characteristics:appearance",
  "5": "characteristics:personality",
  "6": "characteristics:others",
  "7": "possessions",
  "8": "relationships:family",
  "10": "relationships:friend",
  "11": "relationships:others",
  "12": "experiences:past/present",
  "13": "experience:future",
  "14": "beliefs/values"
}
```
### PANDA🐼 Framework
- This framework consists of the following three multiple steps: 1) dialogue labeling, 2) persona-topic mapping, and 3) overuse measurement.
- The current version is implemented with code for ChatGPT for testing. Refined code for utilization on other models will be further provided.
#### Arguments
```python
## Not all arguments are mandatory. 
--task: {'gen', 'ptag', 'ttag', 'eval'}  # mandatory, 'ptag': dialogue labeling, 'ttag': topic mapping, 'eval': measuring overuse
--model_type: {'chatgpt'}  # mandatory
--prompt: {'vanilla', 'cot', 'decom', 'refine'}  # prompt engineering variants, 'decom': task decomposition, 'refine': self-refine method
--hist_num: {2, 4}  # to slice the length of dialogue history, if 2: including 4 turns(utterances), 4: 8 turns.
--input_type: {'h+u', 'u_only'}  # to ablate dialogue history
--start_idx: int (>= 0)  # to debug, start instance index
--end_idx: int (>= 0)  # to debug, end instance index
--input_path: str  # to set input path manually 
--output_path: str  # to set output path manually 
--api_key: str  # to provide api key manually (But, we recommend using "openai_api_key.txt" or the dotenv library.
```
#### 0. 모델의 응답 생성
- Before applying the **PANDA** framework, we generate LLM response in a persona-grounded dialogue generation task. 
```bash
$ scripts/run_main.sh
```
#### 1. Dialogue Labeling
- In this step, automated annotation is conducted on all persona attributes included in each utterance (or model response) by adopting the 'LLM-as-a-Labeler' approach. 
```bash
$ bash scripts/run_dialogue_labeling.sh
```
#### 2. Persona-Topic Mapping
- This step performs a mapping task from a set of persona attributes contained in an utterance (or model response) to a set of dialogue topics.
- It utilizes information that has been previously human-annotated for the mapping task.
```bash
$ bash scripts/run_dialogue_labeling.sh
```
#### 3. Overuse Measurement
- The overuse problem consists of two cases: 'off-topic' and 'excess of quantity.'
- The overuse score (OVS) is calculated based on the dialogue topics extracted in the previous steps.
```bash
$ bash scripts/run_eval.sh
```

## 📖 Citation
```
@inproceedings{kim2024panda,
  title={{PANDA}: Persona Attributes Navigation for Detecting and Alleviating Overuse Problem in Large Language Models},
  author={Kim, Jinsung and Koo, Seonmin and Lim, Heui-Seok},
  booktitle={The 2024 Conference on Empirical Methods in Natural Language Processing},
  year={2024},
  url={https://openreview.net/forum?id=MhxzXcjTka}
}
```

### Misc.
- In addition, our another paper [Where am I? Large Language Models Wandering between Semantics and Structures in Long Contexts](https://openreview.net/forum?id=gY1eSVMx2E&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DEMNLP%2F2024%2FConference%2FAuthors%23your-submissions)), which was accepted to EMNLP 2024 Main, is also a recommended LLM probing paper. 
- This paper addresses the discrepancy between LLMs' different abilities: to answer in a QA task and to select evidence for their responses.
