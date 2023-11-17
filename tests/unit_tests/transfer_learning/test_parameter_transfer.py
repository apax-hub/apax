from apax.transfer_learning import black_list_param_transfer


def test_param_transfer():
    source = {
        "params": {
            "dense": {"w": 1, "b": 2},
            "basis": {"emb": 3},
        }
    }
    target = {
        "params": {
            "dense": {"w": 0, "b": 0},
            "basis": {"emb": 0},
        }
    }
    reinitialize_layers = ["basis"]
    transfered_target = black_list_param_transfer(source, target, reinitialize_layers)

    assert transfered_target["params"]["dense"]["w"] == source["params"]["dense"]["w"]
    assert transfered_target["params"]["dense"]["b"] == source["params"]["dense"]["b"]
    assert transfered_target["params"]["basis"]["emb"] == target["params"]["basis"]["emb"]
