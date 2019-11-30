from pathlib import Path
from mapping import Mapping

DEFALULT_CATEGORIES_FN = str(Path('./gcon/e2e/data/ids_category').resolve())


class Categories(Mapping):

    def __init__(self, fn: str = DEFALULT_CATEGORIES_FN) -> None:
        super(Categories, self).__init__()
        print('load ids_category...')
        try:
            self.load(fn)
        except FileNotFoundError:
            print('File not find. Use defalult categories dictionary.')
            self.load(DEFALULT_CATEGORIES_FN)
