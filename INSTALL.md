# Graphiti-Core 설치 가이드

이 문서는 sprintsolo 조직 구성원들이 graphiti-core 패키지를 설치하는 방법을 안내합니다.

## 사전 요구사항

1. **SSH 키 설정**: GitHub에 SSH 키가 등록되어 있어야 합니다.
2. **조직 접근 권한**: sprintsolo 조직의 멤버여야 합니다.

## SSH 키 설정 확인

```bash
# SSH 키가 GitHub에 등록되어 있는지 확인
ssh -T git@github.com
```

성공적으로 설정되었다면 다음과 같은 메시지가 나타납니다:
```
Hi username! You've successfully authenticated, but GitHub does not provide shell access.
```

## 설치 방법

### 방법 1: pip로 직접 설치

```bash
pip install git+ssh://git@github.com/sprintsolo/graphiti.git
```

### 방법 2: requirements.txt 사용

requirements.txt 파일에 다음 라인을 추가:

```
git+ssh://git@github.com/sprintsolo/graphiti.git
```

그리고 설치:

```bash
pip install -r requirements.txt
```

### 방법 3: 특정 브랜치나 태그 설치

```bash
# 특정 브랜치 설치
pip install git+ssh://git@github.com/sprintsolo/graphiti.git@branch-name

# 특정 태그 설치
pip install git+ssh://git@github.com/sprintsolo/graphiti.git@v0.11.6

# 특정 커밋 설치
pip install git+ssh://git@github.com/sprintsolo/graphiti.git@commit-hash
```

### 방법 4: Poetry 사용 (개발자용)

pyproject.toml에 추가:

```toml
[tool.poetry.dependencies]
graphiti-core = {git = "ssh://git@github.com/sprintsolo/graphiti.git"}
```

또는 명령어로:

```bash
poetry add git+ssh://git@github.com/sprintsolo/graphiti.git
```

## 개발 모드 설치

로컬에서 개발하려면:

```bash
# 저장소 클론
git clone git@github.com:sprintsolo/graphiti.git
cd graphiti

# 개발 모드로 설치
pip install -e .

# 또는 Poetry 사용
poetry install
```

## 문제 해결

### SSH 키 문제

SSH 키가 설정되지 않았다면:

1. SSH 키 생성:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

2. SSH 에이전트에 키 추가:
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

3. GitHub에 공개 키 등록:
```bash
cat ~/.ssh/id_ed25519.pub
```
이 내용을 GitHub Settings > SSH and GPG keys에 추가

### 권한 문제

조직 접근 권한이 없다면 조직 관리자에게 문의하세요.

### HTTPS 대신 SSH 사용하기

Git이 HTTPS를 사용하도록 설정되어 있다면:

```bash
git config --global url."git@github.com:".insteadOf "https://github.com/"
```

## 사용 예시

설치 후 사용:

```python
from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig

# Graphiti 인스턴스 생성
graphiti = Graphiti(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j", 
    neo4j_password="password",
    llm_client=GeminiClient(config=LLMConfig(api_key="your-api-key"))
)
```

## 업데이트

패키지를 최신 버전으로 업데이트:

```bash
pip install --upgrade git+ssh://git@github.com/sprintsolo/graphiti.git
``` 