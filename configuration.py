IO_FOLDER = "data"

DOCS_FILE = "docs.txt"
QUERIES_FILE = "queries.txt"
RELEVANCE_FILE = "qrel.txt"


STOP_WORDS = [" ", "", "abstract", "excerpt", "preface", "summary", "short", "preface", "ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with ", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", " your ", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"]
TERM_SPLIT_CHARACTERS = ["\" ", " \"", "?", ", ", "!", ".\n", ". ", "&", "\n", ";", ":", "...", " - ", "\\", "/", "(", ")", "[", "]"]

ABSTRACT_BEGINNINGS = ["Abstract", "Preface", "Summary", "Short", "Synopsis", "Excerpt"]

# Amount of documents that should be returned
TOP_K_DOCS = 20

# K inside wf_t_d method
K = 2