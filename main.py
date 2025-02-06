from flask import Flask, jsonify, render_template
import os
import random
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# 환경변수 설정
NUM_RECOMMENDATIONS = int(os.getenv("NUM_RECOMMENDATIONS", 5))
HISTORY_FILE = "lotto_history.json"  # 최근 당첨 이력을 저장하는 파일

# 최근 당첨 이력 로드 함수
def load_lotto_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as file:
            return json.load(file)
    return []

# 로또 번호 생성 함수
def generate_lotto_numbers():
    return sorted(random.sample(range(1, 46), 6))

# 최근 당첨 번호 기반 분석 함수
def analyze_lotto_history():
    history = load_lotto_history()
    number_counts = {i: 0 for i in range(1, 46)}
    
    for draw in history:
        for num in draw:
            number_counts[num] += 1
    
    # 가장 많이 나온 숫자 TOP 10 추출
    sorted_numbers = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)
    most_common_numbers = [num for num, _ in sorted_numbers[:10]]
    
    return most_common_numbers

# 상세한 분석을 위한 프롬프트 생성 함수
def generate_lotto_prompt():
    history_data = analyze_lotto_history()
    history_str = ", ".join(map(str, history_data))

    prompt = f"""
    당신은 로또 분석 전문가입니다. 아래 데이터를 기반으로, 로또 번호의 출현 패턴을 분석하고 다음 회차에서 당첨될 가능성이 높은 번호 6개를 추천하세요.
    
    📌 **[분석 데이터]**
    1. **최근 당첨 번호**: {history_str}
    2. **번호 빈도 분석**
       - 가장 많이 나온 번호: (이전 당첨 기록에서 10회 이상 등장한 번호)
       - 가장 적게 나온 번호: (최근 100회 동안 5회 미만 등장한 번호)
    3. **연속 번호 패턴**
       - 연속된 숫자 조합(예: 12-13, 25-26 등)이 자주 등장하는가?
       - 3연속 번호가 등장하는 확률이 높은가?
    4. **홀수/짝수 비율**
       - 최근 10회 동안의 평균 홀짝 비율: (예: 3:3, 4:2 등)
    5. **고/저 비율**
       - 낮은 숫자(1~22)와 높은 숫자(23~45)의 출현 빈도
    6. **끝자리 분석**
       - 같은 끝자리 숫자(예: 5, 15, 25, 35)가 자주 등장하는가?
    7. **보너스 번호 패턴**
       - 보너스 번호와 함께 자주 출현하는 숫자 리스트
    8. **특정 회차 패턴**
       - 5, 10, 20, 50의 배수 회차에서 자주 나온 번호
    
    🎯 **[목표]**
    - 위 분석을 활용하여 가장 가능성이 높은 6개의 숫자를 추천하세요.
    - 추천하는 숫자가 위 분석과 어떻게 관련 있는지 설명하세요.
    
    🔢 **추천 번호 6개를 아래 형식으로 출력하세요:**
    ```
    추천 번호: [12, 23, 34, 7, 8, 41]
    분석 이유: (여기에 설명 작성)
    ```
    """.strip()
    
    return prompt

# LLM을 활용한 추천 분석 함수
def analyze_lotto_pattern(model_name="distilgpt2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    prompt = generate_lotto_prompt()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)  # 🔹 입력 길이 제한 추가
    inputs = {k: v.to(device) for k, v in inputs.items()}  # 🔹 GPU/CPU 적용

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)  # 🔹 최대 출력 길이 제한
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("🔍 LLM 분석 결과:\n", generated_text)

    recommended_numbers = [int(num) for num in generated_text.split() if num.isdigit()]

    if len(recommended_numbers) < 6:
        recommended_numbers += generate_lotto_numbers()

    return sorted(recommended_numbers[:6])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/lotto", methods=["GET"])
def get_lotto_recommendations():
    results = [analyze_lotto_pattern() for _ in range(NUM_RECOMMENDATIONS)]
    return jsonify({"recommended_lotto_numbers": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
