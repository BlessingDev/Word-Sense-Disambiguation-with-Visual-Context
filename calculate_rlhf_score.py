import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# 1. 효율적인 데이터 로딩을 위한 커스텀 데이터셋 클래스
class TextDataset(Dataset):
    def __init__(self, df):
        self.set_df = df
        self.question_list = list()
        self.answer_list = df["generated_sentence"].tolist()
        prompt_template = """Please create a natural sentence using the word '{word}'."""
        
        for idx, row in df.iterrows():
            self.question_list.append(prompt_template.format(word=row["word"]))

    def __len__(self):
        return len(self.set_df)

    def __getitem__(self, idx):
        # 텍스트가 비어있을 경우 예외 처리
        question = self.question_list[idx] if idx < len(self.question_list) else ""
        answer = self.answer_list[idx] if idx < len(self.answer_list) else ""
        return question, answer

def evaluate_rewards_batched(args):
    # 2. 모델 및 토크나이저 로드
    model_name = args.model_checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    # 3. 데이터 준비
    df = pd.read_csv(args.input_csv)
    dataset = TextDataset(df)
    # num_workers를 설정하면 데이터 로딩이 더 빨라집니다 (Windows는 0 권장)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    all_scores = []

    print(f"Batch Size: {args.batch_size} | Total Samples: {len(df)} | Device: {device}")

    # 4. 배치 단위 추론
    with torch.no_grad():
        for batch_questions, batch_answers in tqdm(dataloader):
            # 한 번에 토큰화 (padding=True로 배치 내 길이를 맞춤)
            inputs = tokenizer(
                batch_questions, batch_answers,
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024
            ).to(device)
            
            # 모델 추론
            outputs = model(**inputs)
            # Logits를 리스트로 변환하여 저장
            scores = outputs.logits[:, 0].view(-1).cpu().tolist()
            all_scores.extend(scores)

    # 5. 결과 저장
    df['reward_score'] = all_scores
    df.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 처리가 완료되었습니다. 결과 파일: {args.output_csv}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate reward scores for generated texts using a reward model.")
    parser.add_argument("--model_checkpoint", type=str, default="OpenAssistant/reward-model-deberta-v3-large", help="Hugging Face 모델 체크포인트")
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV file containing generated texts.")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV file to save the results with reward scores.")
    parser.add_argument("--text_column", type=str, default="generated_sentence", help="Column name in the CSV that contains the generated texts.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing texts.")

    args = parser.parse_args()
    evaluate_rewards_batched(args)