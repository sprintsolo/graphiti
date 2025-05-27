# Vertex AI 서비스 계정 키 설정 가이드

이 문서는 Graphiti-Core에서 Vertex AI를 서비스 계정 키로 사용하는 방법을 안내합니다.

## 1. Google Cloud 서비스 계정 생성

### 1.1 Google Cloud Console에서 서비스 계정 생성

1. [Google Cloud Console](https://console.cloud.google.com/)에 접속
2. 프로젝트 선택 또는 새 프로젝트 생성
3. **IAM 및 관리자** > **서비스 계정**으로 이동
4. **서비스 계정 만들기** 클릭
5. 서비스 계정 세부정보 입력:
   - 서비스 계정 이름: `graphiti-vertex-ai`
   - 서비스 계정 ID: `graphiti-vertex-ai`
   - 설명: `Graphiti Vertex AI Service Account`

### 1.2 권한 부여

서비스 계정에 다음 역할을 부여:
- **Vertex AI User** (`roles/aiplatform.user`)
- **ML Developer** (`roles/ml.developer`) (선택사항)

### 1.3 키 생성

1. 생성된 서비스 계정 클릭
2. **키** 탭으로 이동
3. **키 추가** > **새 키 만들기**
4. **JSON** 형식 선택
5. 키 파일 다운로드 (예: `service-account-key.json`)

## 2. 환경 설정

### 방법 1: 환경 변수 사용

```bash
# 서비스 계정 키 파일 경로 설정
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# 프로젝트 ID 설정
export GOOGLE_CLOUD_PROJECT="your-project-id"

# 리전 설정 (선택사항, 기본값: us-central1)
export GOOGLE_CLOUD_LOCATION="us-central1"

# 서비스 계정 키 파일 경로 (선택사항)
export SERVICE_ACCOUNT_KEY_PATH="/path/to/service-account-key.json"
```

### 방법 2: .env 파일 사용

`.env` 파일에 다음 내용 추가:

```env
# Neo4j 설정
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Vertex AI 서비스 계정 설정
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
SERVICE_ACCOUNT_KEY_PATH=/path/to/service-account-key.json

# 모델 설정
LLM_MODEL=gemini-2.5-flash-preview-05-20
EMBEDDING_MODEL=text-multilingual-embedding-002
EMBEDDING_DIM=768
```

## 3. 코드에서 사용

### 3.1 기본 사용법

```python
import json
from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient

# 서비스 계정 키 파일에서 로드
with open('/path/to/service-account-key.json', 'r') as f:
    service_account_key_json = json.load(f)

# LLM 설정
llm_config = LLMConfig(
    model="gemini-2.5-flash-preview-05-20",
    service_account_key_json=service_account_key_json,
    project_id="your-project-id",
    location="us-central1"
)

# Embedder 설정
embedder_config = GeminiEmbedderConfig(
    embedding_model="text-multilingual-embedding-002",
    embedding_dim=768,
    service_account_key_json=service_account_key_json,
    project_id="your-project-id",
    location="us-central1"
)

# Graphiti 초기화
graphiti = Graphiti(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    llm_client=GeminiClient(config=llm_config),
    embedder=GeminiEmbedder(config=embedder_config),
    cross_encoder=GeminiRerankerClient(config=llm_config)
)
```

### 3.2 파일 경로 사용

```python
# 서비스 계정 키 파일 경로 사용
llm_config = LLMConfig(
    model="gemini-2.5-flash-preview-05-20",
    service_account_key_path="/path/to/service-account-key.json",
    project_id="your-project-id",
    location="us-central1"
)

embedder_config = GeminiEmbedderConfig(
    embedding_model="text-multilingual-embedding-002",
    embedding_dim=768,
    service_account_key_path="/path/to/service-account-key.json",
    project_id="your-project-id",
    location="us-central1"
)
```

### 3.3 환경 변수 사용 (권장)

```python
# 환경 변수가 설정되어 있으면 자동으로 감지
llm_config = LLMConfig(
    model="gemini-2.5-flash-preview-05-20"
    # service_account_key_path, project_id, location은 환경 변수에서 자동 로드
)

embedder_config = GeminiEmbedderConfig(
    embedding_model="text-multilingual-embedding-002",
    embedding_dim=768
    # service_account_key_path, project_id, location은 환경 변수에서 자동 로드
)
```

## 4. 검색 예시

```python
from examples.quickstart.search import llm_controlled_search

async def main():
    # ... graphiti 초기화 ...
    
    # 검색 수행
    results = await llm_controlled_search(
        graphiti,
        "삼성전자 직원",
        use_bm25=True,
        use_cosine_similarity=True,
        node_labels=["Person"],
        limit=10
    )
    
    # 결과 출력
    if hasattr(results, "nodes") and results.nodes:
        for node in results.nodes:
            print(f"이름: {node.name}")
            print(f"라벨: {', '.join(node.labels)}")
            print("---")

# 실행
import asyncio
asyncio.run(main())
```

## 5. 문제 해결

### 5.1 인증 오류

```
google.auth.exceptions.DefaultCredentialsError: Could not automatically determine credentials.
```

**해결책:**
1. `GOOGLE_APPLICATION_CREDENTIALS` 환경 변수가 올바른 경로를 가리키는지 확인
2. 서비스 계정 키 파일이 존재하고 읽기 권한이 있는지 확인
3. JSON 파일 형식이 올바른지 확인

### 5.2 권한 오류

```
google.api_core.exceptions.PermissionDenied: 403 Permission denied
```

**해결책:**
1. 서비스 계정에 **Vertex AI User** 역할이 부여되었는지 확인
2. 프로젝트에서 Vertex AI API가 활성화되었는지 확인
3. 올바른 프로젝트 ID를 사용하고 있는지 확인

### 5.3 할당량 오류

```
google.api_core.exceptions.ResourceExhausted: 429 Quota exceeded
```

**해결책:**
1. Google Cloud Console에서 할당량 확인
2. 필요시 할당량 증가 요청
3. 요청 빈도 조절

## 6. 보안 고려사항

1. **서비스 계정 키 파일 보안:**
   - 키 파일을 버전 관리 시스템에 커밋하지 마세요
   - `.gitignore`에 키 파일 패턴 추가: `*.json`, `service-account-*.json`

2. **최소 권한 원칙:**
   - 필요한 최소한의 권한만 부여
   - 정기적으로 권한 검토

3. **키 순환:**
   - 정기적으로 서비스 계정 키 교체
   - 사용하지 않는 키 삭제

## 7. 추가 리소스

- [Google Cloud 서비스 계정 문서](https://cloud.google.com/iam/docs/service-accounts)
- [Vertex AI 문서](https://cloud.google.com/vertex-ai/docs)
- [Google Auth 라이브러리 문서](https://google-auth.readthedocs.io/) 