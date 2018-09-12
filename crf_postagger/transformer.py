def potential_function(pos):
    """Feature transformer
    
    :param list_of_tuple pos: a sentence [(word, tag), (word, tag), ...]
    """
    words, tags = zip(*pos)
    raise NotImplemented