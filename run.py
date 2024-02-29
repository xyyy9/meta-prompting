import time
import logging
import logging.config
import pandas as pd
from dotenv import load_dotenv
import os
from openai import OpenAI
from pandas import ExcelWriter
from pydantic import BaseModel
import json
import google.generativeai as genai



load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai_gptmodel="gpt-4-1106-preview"
openai_gptmodel_func="gpt-4-1106-preview"

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

logging.config.fileConfig('logging_config.ini')
logger = logging.getLogger('gptLogger')

openai_temperature = float(os.getenv("OPENAI_TEMPERATURE"))
exam_excel = os.getenv("EXAM_EXCEL")
results_excel = os.getenv("RESULTS_EXCEL")

class Expert(BaseModel):
    expert_name: str


class GradedAnswer(BaseModel):
    def __init__(self, grade: float, explanation: str):
        self.grade = grade
        self.explanation = explanation

    grade: float
    explanation: str


def load_excel_data() -> pd.DataFrame:
    sheet_name = "F23GEM"
    try:
        data = pd.read_excel(exam_excel, sheet_name=sheet_name)
        logging.info('Data loaded successfully from file %s and sheet %s', exam_excel, sheet_name)
        return data
    except Exception as e:
        logging.error('Failed to load data from file %s and sheet %s', exam_excel, sheet_name)
        logging.error(e)
        data = None
    return data


def get_prompt_config() -> dict:    
    sheet_name = "Prompts"
    try:
        data = pd.read_excel(exam_excel, sheet_name=sheet_name, header=None)
        prompt_config = dict(data.values)
        logging.info('Prompt config loaded successfully from file %s and sheet %s', exam_excel, sheet_name)
        return prompt_config
    except Exception as e:
        logging.error('Failed to load prompt config from file %s and sheet %s', exam_excel, sheet_name)
        logging.error(e)
        return None


def answer_question(system: str, question: str) -> str:
    try:    
        logging.info(f'System: {system}')
        logging.info(f'Question: {question}')
        messages = [
            {'role':'user',
             'parts': [system]}
             ]
        response = model.generate_content(messages)
        messages.append({'role':'model',
                 'parts':[response.text]})
        messages.append({'role':'user',
                 'parts':[question]})
        response = model.generate_content(messages)
        answer = response.text
        # response = client.chat.completions.create(model=openai_gptmodel,
        # max_tokens=4000,
        # temperature=openai_temperature,
        # messages=[
        #     {
        #     "role": "system",
        #     "content": system
        #     },
        #     {
        #     "role": "user",
        #     "content": question
        #     }
        # ])
        # answer = response.choices[0].message.content
        logging.info(f'Answer: {answer}')
        return answer
    except Exception as e:
        logging.error('Failed to answer will try again in 30s', exc_info=True)
        time.sleep(30)
        
        return answer_question(system, question)
    

def grade_question(grader: str, question: str, answer: str, solution: str, prompt: str) -> str:
    try:    
        logging.info("Grading started")
        logging.info(f'System: {grader}')
        logging.info(f'Question: {question}')
        logging.info(f'Answer: {answer}')
        logging.info(f'Solution: {solution}')
        logging.info(f'Prompt: {prompt}')
        grade_prompt = prompt.replace("<Question>", question)
        grade_prompt = grade_prompt.replace("<Solution>", solution)
        grade_prompt = grade_prompt.replace("<Answer>", answer)
        logging.info(f'Final prompt: {grade_question}')
        response = client.chat.completions.create(model=openai_gptmodel,
        max_tokens=4000,
        temperature=openai_temperature,
        messages=[
            {
            "role": "system",
            "content": grader
            },
            {
            "role": "user",
            "content": grade_question
            }
        ])
        answer = response.choices[0].message.content
        logging.info(f'Answer: {answer}')
        return answer
    except Exception as e:
        logging.error('Failed to answer will try again in 30s')
        logging.error(e)
        time.sleep(30)
        
        return grade_question(grader, question, answer, solution, prompt)
    
#Grade question with function calling
def grade_question_f(grader: str, question: str, answer: str, solution: str, prompt: str) -> GradedAnswer:
    try:    
        logging.info("Grading started")
        logging.info(f'System: {grader}')
        logging.info(f'Question: {question}')
        logging.info(f'Answer: {answer}')
        logging.info(f'Solution: {solution}')
        logging.info(f'Prompt: {prompt}')
        grade_question = prompt.replace("<Question>", question)
        grade_question = grade_question.replace("<Solution>", solution)
        grade_question = grade_question.replace("<Answer>", answer)
        logging.info(f'Final prompt: {grade_question}')
        response = client.chat.completions.create(model=openai_gptmodel_func,
        max_tokens=4000,
        temperature=openai_temperature,
        messages=[
            {
            "role": "system",
            "content": grader
            },
            {
            "role": "user",
            "content": grade_question
            }
        ],
            functions=[
            {
                "name": "grade_question",
                "description": "Grade a question based on the answer and solution",
                "parameters": GradedAnswer.schema()
            }
            ],
            function_call={"name": "grade_question"})
        arguments_str = response.choices[0].message.function_call.arguments
        arguments_str = arguments_str.replace("\\", "\\\\")
        graded_answer_json = json.loads(arguments_str, strict=False)
        return float(graded_answer_json["grade"]), graded_answer_json["explanation"]
    except Exception as e:
        logging.error('Failed to answer will try again in 30s')
        logging.error(e, exc_info=True)
        time.sleep(30)
        
        return grade_question(grader, question, answer, solution, prompt)


def get_expert(generic_prof: str, graded_question: str, prompt: str) -> str:    
    try:
        user_prompt = prompt.replace("<Question>", graded_question)
        logger.info(f'User Prompt: {user_prompt}')
        response = client.chat.completions.create(model=openai_gptmodel_func,
        temperature=openai_temperature,
        messages=[
            {"role":"system", "content": generic_prof},
            {"role":"user", "content": user_prompt}
        ],
        functions=[
        {
            "name": "get_named_expert_for_question",
            "description": "Get named expert for answering a question",
            "parameters": Expert.schema()
        }
        ],
        function_call={"name": "get_named_expert_for_question"})
        arguments_str = response.choices[0].message.function_call.arguments
        expert_name = json.loads(arguments_str)
        
        return expert_name.get("expert_name")
    except Exception as e:
        logger.error(f"Failed to get Expert for {graded_question} will try again in 30s")
        logging.exception(e)
        time.sleep(30)
        return get_expert(generic_prof, graded_question, prompt)
    


def main():
    df = load_excel_data()
    prompts = get_prompt_config()

    for index, row in df.iterrows():    
        system_e = row["E"] #Professor
        expert_name_for_question = get_expert(system_e, row["Question"], prompts["Expert"])
        system_qe = prompts["QE"].replace("<Full Name>", expert_name_for_question)
    
        row["QE"] = system_qe ## added for row F
        
        question = row["Question"]
        
        # E+ZS+CoT: Professor + Zero Shot + Cot
        zs_cot_prompt = prompts["ZS+CoT"]
        zs_cot_question = zs_cot_prompt.replace("<Question>", question)  # Zero Shot + Cot question

        #E+ZS: Professor + Zero Shot
        # logging.info("starting E+ZS")
        # answer = answer_question(system_e, question)
        # row["E+ZS"] = answer
        # grading, explanation = grade_question_f(grader=system_e, question=question, answer=row["E+ZS"], solution=row["Solution"], prompt=prompts["Grading"])
        # row["E+ZS Grade"] = grading
        # row["E+ZS Explanation"] = explanation
        

        # Add "Let's think step by step" at the end of the prompt          
        # logging.info("starting E+ZS+CoT")
        # row["E+ZS+CoT"] = answer_question(system_e, zs_cot_question)
        # grading, explanation = grade_question_f(grader=system_e, question=zs_cot_question, answer=row["E+ZS+CoT"], solution=row["Solution"], prompt=prompts["Grading"])
        # row["E+ZS+CoT Grade"] = grading
        # row["E+ZS+CoT Explanation"] = explanation

        #QE+ZS: Named Professor + Zero Shot
        # logging.info("starting QE+ZS")
        # row["QE+ZS"] = answer_question(system_qe, question)
        # grading, explanation = grade_question_f(grader=system_e, question=question, answer=row["QE+ZS"], solution=row["Solution"], prompt=prompts["Grading"])
        # row["QE+ZS Grade"] = grading
        # row["QE+ZS Explanation"] = explanation

        #QE+ZS+CoT: Named Professor + Zero Shot + Cot
        logging.info("starting QE+ZS+CoT")
        row["QE+ZS+CoT"] = answer_question(system_qe, zs_cot_question)

        #skipping grading
        # grading, explanation = grade_question_f(grader=system_e, question=zs_cot_question, answer=row["QE+ZS+CoT"], solution=row["Solution"], prompt=prompts["Grading"])
        # row["QE+ZS+CoT Grade"] = grading
        # row["QE+ZS+CoT Explanation"] = explanation
        
        row["Updated At"] = pd.Timestamp.now()
        
        df.loc[index] = row    

        with ExcelWriter(results_excel, engine='openpyxl', mode='a') as writer:
            if 'F23GEM' in writer.book.sheetnames:
                std = writer.book['F23GEM']
                writer.book.remove(std)
            df.to_excel(writer, sheet_name='F23GEM', index=False)
            writer.book.save(results_excel)

    logging.info(f'Finished answering and grading {len(df)} questions')



if (__name__) == '__main__':
    main()
