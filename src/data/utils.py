from typing import List
from tqdm import tqdm


def yield_batches(input_list: List[object], batch_size=12) -> List[object]:
    for i in tqdm(range(0, len(input_list), batch_size)):
        yield input_list[i: i + batch_size]

