import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import requests
import io
import sys
import torch

def download_coco_image(image_id, save_path=None):
    image_urls = [
        "http://images.cocodataset.org/train2017/000000115681.jpg",
        "http://images.cocodataset.org/val2017/000000115681.jpg",
        "http://images.cocodataset.org/test2017/000000115681.jpg",
        "http://images.cocodataset.org/train2014/000000115681.jpg",
        "http://images.cocodataset.org/val2014/000000115681.jpg"
    ]
    
    for image_url in image_urls:
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            if save_path:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                print(f"이미지 다운로드 완료: {save_path}")
            
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            print(f"이미지를 찾았습니다: {image_url}")
            return image
        except requests.exceptions.RequestException:
            continue
    
    print(f"이미지 다운로드 실패: 이미지 ID {image_id}를 모든 COCO 세트에서 찾을 수 없습니다.")
    print("시도한 URL들:")
    for image_url in image_urls:
        print(f"  {image_url}")
    return None

def get_query_tokens(model_name="Salesforce/blip2-opt-2.7b"):
    print(f"원본 모델에서 query_tokens 로드 중: {model_name}")
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    query_tokens = model.query_tokens.detach().cpu().numpy()
    if len(query_tokens.shape) == 3:
        query_tokens = query_tokens.squeeze(0)
    print(f"query_tokens shape: {query_tokens.shape}")
    return query_tokens

def preprocess_image(image, processor):
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.squeeze(0).numpy()
    return pixel_values.astype(np.float32)

def prepare_inputs(image, prompt, processor, query_tokens):
    pixel_values = preprocess_image(image, processor)
    
    text_inputs = processor(text=prompt, return_tensors="pt")
    input_ids = text_inputs.input_ids.squeeze(0).numpy().astype(np.int64)
    attention_mask = text_inputs.attention_mask.squeeze(0).numpy().astype(np.int64)
    
    if len(query_tokens.shape) == 3:
        query_tokens = query_tokens.squeeze(0)
    
    expected_input_length = 1
    if len(input_ids) != expected_input_length:
        print(f"경고: input_ids 길이가 {len(input_ids)}입니다. {expected_input_length}로 조정합니다.")
        input_ids = input_ids[:expected_input_length]
        attention_mask = attention_mask[:expected_input_length]
    
    vision_attention_mask = np.ones(257, dtype=np.float32)
    query_attention_mask = np.ones(32, dtype=np.float32)
    
    pixel_values = np.expand_dims(pixel_values, axis=0)
    vision_attention_mask = np.expand_dims(vision_attention_mask, axis=0)
    query_attention_mask = np.expand_dims(query_attention_mask, axis=0)
    query_tokens = np.expand_dims(query_tokens, axis=0)
    input_ids = np.expand_dims(input_ids, axis=0)
    attention_mask = np.expand_dims(attention_mask, axis=0)
    
    return {
        "pixel_values": pixel_values,
        "vision_attention_mask": vision_attention_mask,
        "query_attention_mask": query_attention_mask,
        "query_tokens": query_tokens.astype(np.float32),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

def infer_triton(inputs, model_name="blip2_ensemble", url="localhost:8000", model_version="1"):
    client = httpclient.InferenceServerClient(url=url)
    
    triton_inputs = []
    for name, data in inputs.items():
        dtype = "FP32" if data.dtype == np.float32 else "INT64"
        triton_input = httpclient.InferInput(name, data.shape, dtype)
        triton_input.set_data_from_numpy(data)
        triton_inputs.append(triton_input)
    
    output = httpclient.InferRequestedOutput("logits")
    
    print(f"Triton 서버에 추론 요청 전송 중... (모델: {model_name})")
    response = client.infer(model_name, model_version=model_version, inputs=triton_inputs, outputs=[output])
    
    logits = response.as_numpy("logits")
    print(f"추론 완료. logits shape: {logits.shape}")
    return logits

def decode_logits_greedy(logits, processor, input_ids, max_new_tokens=50):
    """
    Greedy decoding을 사용하여 logits에서 캡션 생성
    
    Args:
        logits: [batch_size, seq_len, vocab_size] 형태의 logits
        processor: BLIP2Processor
        input_ids: 입력 프롬프트 토큰 IDs [batch_size, prompt_len]
        max_new_tokens: 최대 생성 토큰 수
    
    Returns:
        생성된 캡션 텍스트
    """
    logits = torch.from_numpy(logits)
    input_ids = torch.from_numpy(input_ids)
    
    batch_size = logits.shape[0]
    seq_len = logits.shape[1]
    
    generated_ids = input_ids.clone()
    
    prompt_length = input_ids.shape[-1]
    
    print(f"디코딩 정보: batch_size={batch_size}, seq_len={seq_len}, prompt_length={prompt_length}")
    
    if seq_len > 0:
        last_input_idx = seq_len - 1
    else:
        last_input_idx = 0
    
    next_token_logits = logits[0, last_input_idx]
    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
    
    generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
    
    print(f"첫 번째 생성 토큰 ID: {next_token_id.item()}")
    print(f"첫 번째 생성 토큰 텍스트: {processor.decode(next_token_id[0], skip_special_tokens=True)}")
    
    generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    print(f"전체 생성 텍스트: {generated_text}")
    
    return generated_text

def generate_caption(image, processor, query_tokens, triton_url="localhost:8000", prompt="a photo of", max_new_tokens=50):
    """
    Triton 서버를 사용하여 이미지 캡션 생성 (autoregressive)
    
    Args:
        image: PIL Image
        processor: BLIP2Processor
        query_tokens: query_tokens numpy array
        triton_url: Triton 서버 URL
        prompt: 프롬프트 텍스트
        max_new_tokens: 최대 생성 토큰 수
    
    Returns:
        생성된 캡션 텍스트
    """
    print("\n이미지 캡셔닝 시작...")
    print(f"프롬프트: '{prompt}'")
    print(f"최대 토큰 수: {max_new_tokens}")
    
    # 초기 입력 준비
    inputs = prepare_inputs(image, prompt, processor, query_tokens)
    generated_ids = inputs["input_ids"].copy()
    
    print("Triton 서버에 autoregressive 추론 시작...")
    
    eos_token_id = processor.tokenizer.eos_token_id if hasattr(processor.tokenizer, 'eos_token_id') else None
    
    for step in range(max_new_tokens):
        # 디버깅: 현재 입력 정보 출력
        current_input_ids = inputs["input_ids"][0] if len(inputs["input_ids"].shape) > 1 else inputs["input_ids"]
        current_text = processor.decode(current_input_ids, skip_special_tokens=True)
        print(f"\n[Step {step + 1}] 입력 정보:")
        print(f"  input_ids shape: {inputs['input_ids'].shape}")
        print(f"  input_ids 값: {current_input_ids.tolist()}")
        print(f"  입력 텍스트: '{current_text}'")
        
        # 현재 입력으로 추론
        logits = infer_triton(inputs, url=triton_url)
        
        # 디버깅: logits 정보 출력
        print(f"  logits shape: {logits.shape}")
        
        # logits shape: (1, seq_len, 50304)
        # 마지막 위치의 logits 사용하여 다음 토큰 예측
        next_token_logits = logits[0, -1, :]  # (50304,)
        
        # 디버깅: top-k 토큰 확인
        top_k = 5
        top_k_indices = np.argsort(next_token_logits)[-top_k:][::-1]
        top_k_probs = np.exp(next_token_logits[top_k_indices]) / np.sum(np.exp(next_token_logits))
        print(f"  Top-{top_k} 후보 토큰:")
        for i, (idx, prob) in enumerate(zip(top_k_indices, top_k_probs)):
            token_text = processor.decode([idx], skip_special_tokens=True)
            print(f"    {i+1}. ID={idx}, prob={prob:.4f}, text='{token_text}'")
        
        next_token_id = np.argmax(next_token_logits)
        
        # 생성된 토큰 추가
        generated_ids = np.concatenate([generated_ids, np.array([[next_token_id]])], axis=1)
        
        # 디버깅: 생성된 시퀀스 전체 확인
        generated_text_so_far = processor.decode(generated_ids[0], skip_special_tokens=True)
        print(f"  생성된 토큰 ID: {next_token_id}, 텍스트: '{processor.decode([next_token_id], skip_special_tokens=True)}'")
        print(f"  현재까지 전체 생성 텍스트: '{generated_text_so_far}'")
        
        # 종료 토큰 체크
        if eos_token_id is not None and next_token_id == eos_token_id:
            print(f"  EOS 토큰 생성됨. 생성 중단.")
            break
        
        # 다음 입력 준비 (생성된 모든 토큰 사용)
        inputs["input_ids"] = generated_ids
        inputs["attention_mask"] = np.ones_like(inputs["input_ids"])
    
    generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    print(f"\n전체 생성된 캡션: {generated_text}")
    
    return generated_text

def main():
    image_id = 115681
    prompt = "a photo of"
    triton_url = "localhost:8000"
    
    if len(sys.argv) > 1:
        image_id = int(sys.argv[1])
    if len(sys.argv) > 2:
        prompt = sys.argv[2]
    if len(sys.argv) > 3:
        triton_url = sys.argv[3]
    
    print("=" * 80)
    print("BLIP2-OPT-2.7B Triton 추론 샘플")
    print("=" * 80)
    print(f"COCO 이미지 ID: {image_id}")
    print(f"프롬프트: {prompt}")
    print(f"Triton 서버: {triton_url}")
    print("=" * 80)
    
    print("\n1. 이미지 다운로드 중...")
    image = download_coco_image(image_id)
    if image is None:
        print("이미지 다운로드 실패. 종료합니다.")
        return
    
    print(f"이미지 크기: {image.size}")
    
    print("\n2. 프로세서 로드 중...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    print("\n3. query_tokens 로드 중...")
    query_tokens = get_query_tokens()
    
    print("\n4. 입력 데이터 준비 중...")
    inputs = prepare_inputs(image, prompt, processor, query_tokens)
    
    print("입력 텐서 정보:")
    for name, data in inputs.items():
        print(f"  {name}: shape={data.shape}, dtype={data.dtype}")
    
    print("\n5. 이미지 캡셔닝 수행 중...")
    try:
        caption = generate_caption(
            image=image,
            processor=processor,
            query_tokens=query_tokens,
            triton_url=triton_url,
            prompt=prompt,
            max_new_tokens=50
        )
    except Exception as e:
        print(f"캡셔닝 실패: {e}")
        import traceback
        traceback.print_exc()
        print("\nTriton 서버가 실행 중인지 확인하세요:")
        print("  docker compose up triton")
        return
    
    print("\n" + "=" * 80)
    print("이미지 캡셔닝 결과")
    print("=" * 80)
    print(f"생성된 캡션: {caption}")
    print("=" * 80)

if __name__ == "__main__":
    main()

