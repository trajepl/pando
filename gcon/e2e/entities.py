from pathlib import Path
from mapping import Mapping

DEFALULT_ENTITES_FN = str(Path('./gcon/e2e/data/ids_page').resolve())


class Entities(Mapping):

    def __init__(self, fn: str = DEFALULT_ENTITES_FN) -> None:
        super(Entities, self).__init__()
        print('load ids_pages...')
        try:
            self.load(fn)
        except FileNotFoundError:
            print('File not find. Use defalult entities dictionary.')
            self.load(DEFALULT_ENTITES_FN)
