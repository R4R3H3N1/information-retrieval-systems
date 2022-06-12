import os
import indexer
import configuration

# --------------------------------------------------------------------------- #
if __name__ == '__main__':

    i = indexer.Index(os.path.join(os.getcwd(), "data", configuration.DOCS_FILE))

    print(len(i.dictionary.items()))
