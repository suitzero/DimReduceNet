# Mathematical World Model Roadmap

## 1. 비전 및 핵심 개념 (Vision & Core Concepts)

현재의 월드 모델(World Model)들은 주로 자연어(Natural Language) 프롬프트에 의존하고 있습니다. 하지만 인간의 자연어는 본질적으로 모호성(Ambiguity)을 내포하고 있어, 복잡한 물리 법칙이나 기하학적 형태, 그리고 정밀한 인과관계를 정확하게 기술하기 어렵습니다.

이 프로젝트는 **자연어 대신 Shadertoy와 같은 간단하고 명확한 수학적 언어(Mathematical Language)를 기반으로 하는 월드 모델**을 구축하는 것을 목표로 합니다.

### 핵심 기능: 완벽한 양방향 변환 (Bidirectional Translation)
*   **수식 $\rightarrow$ 영상/동영상 (Forward Rendering):** 명확한 수학적 수식을 입력하면 오차 없이 정확한 영상이나 동영상이 렌더링되어야 합니다.
*   **영상/동영상 $\rightarrow$ 수식 (Backward / Symbolic Discovery):** 영상이나 동영상을 입력하면 그 이면에 존재하는 가장 간결하고 정확한 수학적 수식(규칙)을 도출해냅니다.
*   **유기적 연결:** 수식을 바탕으로 예측 영상을 만들거나, 다시 그 영상을 다른 형태의 수식이나 시뮬레이션으로 자유롭게 넘나들 수 있는 환경을 만듭니다.

## 2. 기존 확률적 모델의 한계 인식 (Recognizing Limitations)
*   **확률론적 생성의 한계:** 기존 딥러닝 기반의 생성 모델(Diffusion, 대규모 LLM 등)은 확률적(Probabilistic) 추론을 수행하므로, 영상이나 수식이 100% "정확하게(Exactly)" 떨어지게 만들지 못하는 치명적인 단점이 있습니다 (예: 프레임 간의 비일관성, 물리 법칙 붕괴).
*   **해결 방향:** 신경망(NN)과 확률 모델을 '탐색과 차원 축소'의 목적으로만 활용하고, 최종 결과물은 결정론적(Deterministic)인 수학적 함수(예: GLSL 셰이더 코드, 최단 길이의 수식)로 강제하여 절대적인 정확성과 압축률을 보장해야 합니다.

---

## 3. MVP (Minimum Viable Product) 정의

가장 먼저 달성해야 할 최소 기능 제품(MVP)은 복잡성을 최소화한 상태에서 **양방향 파이프라인의 가능성과 절대적 정확성을 증명**하는 것입니다.

### MVP 목표: 정적 2D/3D 원시 기하학과 단일 시간 변수 모델링
1.  **제한된 데이터셋 설정:**
    *   2D/3D 기본 도형 (원, 구, 사각형 등) 및 단순 수학 함수 (포물선, 사인파).
    *   시간 `t`에 따른 단순한 선형 이동(Translation) 및 회전(Rotation).
2.  **Forward (수식 $\rightarrow$ 영상):**
    *   수식(예: SDF, `length(p) - r`)을 해석하여 픽셀 이미지 또는 Shadertoy 형태의 GLSL 코드로 즉시 렌더링.
3.  **Backward (영상 $\rightarrow$ 수식 / Vision-to-Code):**
    *   렌더링된 영상을 입력받아 차원 축소(DimReduceNet) 및 진화 알고리즘/LLM 심볼릭 탐색을 통해, 원본 수식을 가장 짧고 간결한 형태(Shortest Function)로 100% 일치시켜 복원해 내는 검증 파이프라인.

---

## 4. 단계별 로드맵 (Phased Roadmap)

### Phase 1: MVP 구축 (정적 2D/3D 기하학 기저 탐색)
*   **목표:** 단순한 도형에 대한 영상 $\leftrightarrow$ 수식 파이프라인 기반 완성.
*   **주요 작업:**
    *   다양한 1D/2D 수학 함수 및 기하학 데이터를 생성하는 `Synthetic Data Factory` 구축.
    *   영상 데이터를 입력받아 압축된 잠재 공간으로 변환하는 `DimReduceNet` 프로토타입 설계.
    *   특징(Feature)을 기반으로 가장 짧은 수식을 찾아내는 **LLM-Guided Symbolic Regression** (유전 프로그래밍 결합) 구현.
    *   결과물을 GLSL/Python 코드로 변환 및 실행 가능한 형태로 출력.

### Phase 2: 동적 물리 시뮬레이션 도입 (시간 변수 `t`와 미분)
*   **목표:** 시간의 흐름(동영상)을 수학적 함수로 압축 (Video $\leftrightarrow$ Math).
*   **주요 작업:**
    *   함수 매개변수에 시간 축 `t`를 추가하여 동적 수식 탐색(Dynamic Symbolic Regression).
    *   튕기는 공, 진동 운동 등 단순한 물리 현상이 담긴 비디오를 궤적 및 물리 방정식으로 역산.
    *   도출된 규칙(함수)을 통해 미래의 프레임 예측(Future Frame Prediction).

### Phase 3: 복잡한 씬 구성 및 CSG (Constructive Solid Geometry)
*   **목표:** 다수의 객체와 복잡한 형태가 결합된 렌더링 씬 모델링.
*   **주요 작업:**
    *   기본 도형뿐만 아니라 합집합(Union), 교집합(Intersection), 차집합(Difference) 등 조합 연산자를 탐색 트리(Function Set)에 추가.
    *   복잡한 3D 씬 영상을 여러 개의 독립된 수학적 함수들의 결합으로 쪼개서 분해.
    *   SDF 기반 레이마칭(Raymarching) 시스템 최적화 코드 자동 생성.

### Phase 4: 궁극의 수학적 월드 모델 (Full Mathematical World Model)
*   **목표:** 복잡한 현실 데이터(실사 비디오, 복잡한 인과관계)를 수학적 언어로 완벽히 이해하고 시뮬레이션.
*   **주요 작업:**
    *   조명, 그림자, 질감 등 복잡한 렌더링 방정식을 탐색 범위 내에 포함.
    *   자연어 프롬프트의 모호성을 완전히 배제하고, 순수 수학적 셰이더 기반으로 세계의 작동 원리를 표현하고 제어하는 궁극의 시뮬레이터 완성.
