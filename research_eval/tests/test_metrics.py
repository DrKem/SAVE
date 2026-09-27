from metrics import evaluate

def test_perfect_single_detection():
    r=[{
        "ground_truth":[{"class_id":0,"box":[0,0,10,10]}],
        "predictions":[{"class_id":0,"score":.9,"box":[0,0,10,10]}],
        "latency_ms":5,
    }]
    m=evaluate(r)
    assert abs(m["precision"]-1)<1e-12
    assert abs(m["recall"]-1)<1e-12
    assert abs(m["mAP50"]-1)<1e-12
    assert abs(m["mAP50_95"]-1)<1e-12
    assert abs(m["latency_ms"]-5)<1e-12

def test_precision_penalizes_false_positive():
    r=[{
        "ground_truth":[{"class_id":0,"box":[0,0,10,10]}],
        "predictions":[
            {"class_id":0,"score":.9,"box":[0,0,10,10]},
            {"class_id":0,"score":.8,"box":[20,20,30,30]},
        ],
        "latency_ms":5,
    }]
    m=evaluate(r)
    assert abs(m["precision"]-.5)<1e-12
    assert abs(m["recall"]-1)<1e-12
