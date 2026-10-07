from gorkha.join import name_score, norm_name, phonetic_key


def test_norm_name_removes_the_type_suffix():
    assert norm_name("Champadevi Rural Municipality") == "champadevi"
    assert norm_name("Panauti Municipality") == "panauti"
    assert norm_name("Hetauda Sub-Metropolitian City") == "hetauda"
    assert norm_name("Hetauda Sub Metropolitan City") == "hetauda"
    assert norm_name("Bharatpur Metropolitan City") == "bharatpur"


def test_norm_name_keeps_letters_only():
    assert norm_name("  Chautara  Sangachok Gadhi ") == "chautarasangachokgadhi"
    assert norm_name("Khijidemba-02") == "khijidemba"
    assert norm_name(None) == ""
    assert norm_name(float("nan")) == ""


def test_phonetic_key_merges_spelling_variants():
    assert phonetic_key("sindhupalchowk") == phonetic_key("sindhupalchok")
    assert phonetic_key("indrawoti") != ""


def test_name_score():
    assert name_score("Panauti Municipality", "Panauti") == 100
    assert name_score("Choutara Sangachowkgadhi Municipality", "Chautara Sangachok Gadhi") > 90
    assert name_score("Langtang National Park", "Likhu") < 50
    assert name_score("Likhu Rural Municipality", None) == 0
