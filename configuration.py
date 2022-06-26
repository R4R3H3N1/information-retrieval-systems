IO_FOLDER = "data"

DOCS_FILE = "docs.txt"
QUERIES_FILE = "queries.txt"
RELEVANCE_FILE = "qrel.txt"


STOP_WORDS = [" ", "", "abstract", "excerpt", "preface", "summary", "short", "preface", "ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with ", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", " your ", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"]
TERM_SPLIT_CHARACTERS = ["\" ", " \"", "?", ", ", "!", ".\n", ". ", "&", "\n", ";", ":", "...", " - ", "\\", "/", "(", ")", "[", "]"]

ABSTRACT_BEGINNINGS = ["Abstract", "Preface", "Summary", "Short", "Synopsis", "Excerpt"]

# Amount of documents that should be returned
TOP_K_DOCS = 10

# K inside wf_t_d method
K = 2

MODEL = 'VEC'

# Latent Semantic Indexing latent space dimensions
LSI_D = 50

LOGGING = False

CAlC_elevenAP = True

USE_EVAL_QUERIES = True
EVAL_QUERIES = ['PLAIN-121', 'PLAIN-1021', 'PLAIN-15', 'PLAIN-145', 'PLAIN-1336']

#EVAL_QUERIES = ['PLAIN-121']
# PLAIN-145 fukushima
#PLAIN-1021 diabetes


# for 11 AP
#EVAL_QUERIES = ['PLAIN-121', 'PLAIN-1021', 'PLAIN-15', 'PLAIN-145', 'PLAIN-1336', 'PLAIN-3', 'PLAIN-4', 'PLAIN-5', 'PLAIN-6','PLAIN-7', 'PLAIN-8', 'PLAIN-9',
#                'PLAIN-10', 'PLAIN-1016', 'PLAIN-2093', 'PLAIN-2094', 'PLAIN-2107', 'PLAIN-2108','PLAIN-2114','PLAIN-2115','PLAIN-2116','PLAIN-2117' ]

