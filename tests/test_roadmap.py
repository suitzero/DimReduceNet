import pytest
from src.roadmap import MVP, DynamicSimulation, CSGModeling, FullWorldModel, Pipeline

def test_mvp_phase():
    phase = MVP()
    assert phase.name == "Phase 1: MVP 구축 (정적 2D/3D 기하학 기저 탐색)"
    assert phase.status == "Not Started"

    result = phase.execute()
    assert result is True
    assert phase.status == "Completed"

def test_dynamic_simulation_phase():
    phase = DynamicSimulation()
    assert phase.name == "Phase 2: 동적 물리 시뮬레이션 도입 (시간 변수 t와 미분)"
    assert phase.status == "Not Started"

    result = phase.execute()
    assert result is True
    assert phase.status == "Completed"

def test_csg_modeling_phase():
    phase = CSGModeling()
    assert phase.name == "Phase 3: 복잡한 씬 구성 및 CSG (Constructive Solid Geometry)"
    assert phase.status == "Not Started"

    result = phase.execute()
    assert result is True
    assert phase.status == "Completed"

def test_full_world_model_phase():
    phase = FullWorldModel()
    assert phase.name == "Phase 4: 궁극의 수학적 월드 모델 (Full Mathematical World Model)"
    assert phase.status == "Not Started"

    result = phase.execute()
    assert result is True
    assert phase.status == "Completed"

def test_pipeline_integration():
    pipeline = Pipeline()
    assert len(pipeline.phases) == 4

    results = pipeline.run_all()
    assert len(results) == 4

    for name, success in results:
        assert success is True

    # Check that all phases are now completed
    for phase in pipeline.phases:
        assert phase.status == "Completed"
