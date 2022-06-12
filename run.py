import os
import indexer
import configuration
import query_processing


def start_query(index: indexer.Index):

    while True:
        user_input = input("Enter query:")
        if user_input == "exit()":
            break
        result = query_processing.execute_query(index, user_input)
        print(f"Result: {result}")


# --------------------------------------------------------------------------- #
if __name__ == '__main__':

    # Create Index
    i = indexer.Index(os.path.join(os.getcwd(), "data", configuration.DOCS_FILE))
    # Start query
    start_query(i)

