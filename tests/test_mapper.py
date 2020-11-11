from tensorflow_similarity.mappers import MemoryMapper

def test_memory_mapper():
    a = {'d': 1}
    b = {'d': 2}
    mapper = MemoryMapper()

    # insert 3 elts
    a_idx = mapper.add(a)
    b_idx = mapper.add(b)
    a2_idx = mapper.add(a)

    # sanity checks
    assert a_idx != b_idx
    assert a_idx != a2_idx
    assert isinstance(a_idx, int)
    assert isinstance(b_idx, int)
    assert isinstance(a2_idx, int)

    # get back three elements
    assert mapper.get(a_idx) == a
    assert mapper.get(b_idx) == b
    assert mapper.get(a2_idx) == a