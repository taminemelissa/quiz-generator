from typing import List
from tqdm import tqdm


def yield_batches(input_list: List[object], batch_size=12) -> List[object]:
    """
    :param input_list: a list that we want to break down into batches
    :param batch_size: the size of one batch
    :return:
    """
    for i in tqdm(range(0, len(input_list), batch_size)):
        yield input_list[i: i + batch_size]

