from .segtoseg_kd import SegToSegKD
#from .kd_losses import KDWithLoss
from .transtocnn import TransToCNN_KD

# 사용할 수 있는 KD 엔진들을 등록
KD_ENGINE_REGISTRY = {
    "segtoseg": SegToSegKD,
    #"kd_losses": KDWithLoss,
    "transtocnn": TransToCNN_KD
    # 여기에 새로운 KD 엔진(예: "attention": AttentionKD)을 추가하면 됨
}


def create_kd_engine(config, teacher, student):
    """
    config 파일의 내용을 바탕으로 적절한 KD 엔진 객체를 생성하여 반환합니다.
    """
    engine_name = config.ENGINE_NAME
    engine_params = config.ENGINE_PARAMS

    if engine_name not in KD_ENGINE_REGISTRY:
        raise ValueError(f"Unknown KD Engine: {engine_name}. Available engines: {list(KD_ENGINE_REGISTRY.keys())}")

    engine_class = KD_ENGINE_REGISTRY[engine_name]

    # teacher, student 모델과 함께 파라미터를 전달하여 엔진 객체 생성
    kd_engine = engine_class(teacher=teacher, student=student, **engine_params)

    return kd_engine