from datasets import load_dataset
from sacremoses import MosesTokenizer
from collections import Counter
import math
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt

def calculate_conditional_entropy(unigram_freq, bigram_freq):

    # Count occurrences of each previous token (context)
    context_freq = Counter()

    for (previous, current), count in bigram_freq.items():
        context_freq[previous] += count

    # Calculate conditional entropy
    entropy = 0

    total_bigrams = sum(bigram_freq.values())

    for (previous, current), count in bigram_freq.items():

        p_xy = count / total_bigrams
        p_y = count / context_freq[previous]

        entropy -= p_xy * math.log2(p_y)

    return entropy

def calculate_ngram_probabilities(unigram_freq, bigram_freq, trigram_freq):

    # Unigram probabilities
    unigram_probs = {
        word: count / sum(unigram_freq.values())
        for word, count in unigram_freq.items()
    }

    # Conditional bigram probabilities: P(w2 | w1)
    bigram_probs = {
        (w1, w2): count / unigram_freq[w1]
        for (w1, w2), count in bigram_freq.items()
    }

    # Conditional trigram probabilities: P(w3 | w1, w2)
    trigram_probs = {
        (w1, w2, w3): count / bigram_freq[(w1, w2)]
        for (w1, w2, w3), count in trigram_freq.items()
    }

    return unigram_probs, bigram_probs, trigram_probs

def get_statistics(tokens):
    # Unigram frequencies
    unigram_freq = Counter(tokens)

    # Bigram frequencies
    bigram_freq = Counter(
        (tokens[i], tokens[i+1])
        for i in range(len(tokens)-1)
    )

    # Trigram frequencies
    trigram_freq = Counter(
        (tokens[i], tokens[i+1], tokens[i+2])
        for i in range(len(tokens)-2)
    )

    stats = {
        "data_size": len(tokens),
        "unique_unigrams": len(unigram_freq),
        "unique_bigrams": len(bigram_freq),
        "unique_trigrams": len(trigram_freq)
    }

    return unigram_freq, bigram_freq, trigram_freq, stats

# Load datasets
ds_en = load_dataset("ufal/npfl147", "en")
ds_cz = load_dataset("ufal/npfl147", "cs")
ds_de = load_dataset("ufal/npfl147", "de")

# Initialize tokenizers for each language
mt_en = MosesTokenizer(lang='en')
mt_cs = MosesTokenizer(lang='cs')
mt_de = MosesTokenizer(lang='de')

# Tokenize and concatenate all documents
tokens_en = []
tokens_cs = []
tokens_de = []

for example in ds_en['train']:
    tokens_en.extend(mt_en.tokenize(example['text']))

for example in ds_cz['train']:
    tokens_cs.extend(mt_cs.tokenize(example['text']))

for example in ds_de['train']:
    tokens_de.extend(mt_de.tokenize(example['text']))

unigrams_en, bigrams_en, trigrams_en, stats_en = get_statistics(tokens_en)
unigrams_cs, bigrams_cs, trigrams_cs, stats_cs = get_statistics(tokens_cs)
unigrams_de, bigrams_de, trigrams_de, stats_de = get_statistics(tokens_de)

entropy_en = calculate_conditional_entropy(unigrams_en, bigrams_en)
entropy_cs = calculate_conditional_entropy(unigrams_cs, bigrams_cs)
entropy_de = calculate_conditional_entropy(unigrams_de, bigrams_de)

results = [
    ["English", len(unigrams_en), len(bigrams_en), len(tokens_en), entropy_en],
    ["Czech", len(unigrams_cs), len(bigrams_cs), len(tokens_cs), entropy_cs],
    ["German", len(unigrams_de), len(bigrams_de), len(tokens_de), entropy_de]
]

headers = ["Language", "Unique tokens", "Unique bigrams", "Data size", "Entropy (bits)"]

print(f"{headers[0]:<12} {headers[1]:<16} {headers[2]:<17} {headers[3]:<15} {headers[4]:<15}")

for row in results:
    print(f"{row[0]:<12} {row[1]:<16} {row[2]:<17} {row[3]:<15} {row[4]:.4f}")

#Czech corpus has a noticeably lower conditional entropy than the others. 
#In our datasets, the next word in Czech is more predictable given the previous one than in German or English

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
# Tokenize and concatenate all documents
tokens_en = []
tokens_cs = []
tokens_de = []

for example in ds_en['train']:
    tokens_en.extend(tokenizer.tokenize(example['text']))

for example in ds_cz['train']:
    tokens_cs.extend(tokenizer.tokenize(example['text']))

for example in ds_de['train']:
    tokens_de.extend(tokenizer.tokenize(example['text']))

unigrams_en, bigrams_en, trigrams_en, stats_en = get_statistics(tokens_en)
unigrams_cs, bigrams_cs, trigrams_cs, stats_cs = get_statistics(tokens_cs)
unigrams_de, bigrams_de, trigrams_de, stats_de = get_statistics(tokens_de)

entropy_en = calculate_conditional_entropy(unigrams_en, bigrams_en)
entropy_cs = calculate_conditional_entropy(unigrams_cs, bigrams_cs)
entropy_de = calculate_conditional_entropy(unigrams_de, bigrams_de)

results = [
    ["English", len(unigrams_en), len(bigrams_en), len(tokens_en), entropy_en],
    ["Czech", len(unigrams_cs), len(bigrams_cs), len(tokens_cs), entropy_cs],
    ["German", len(unigrams_de), len(bigrams_de), len(tokens_de), entropy_de]
]

headers = ["Language", "Unique tokens", "Unique bigrams", "Data size", "Entropy (bits)"]

print(f"{headers[0]:<12} {headers[1]:<16} {headers[2]:<17} {headers[3]:<15} {headers[4]:<15}")

for row in results:
    print(f"{row[0]:<12} {row[1]:<16} {row[2]:<17} {row[3]:<15} {row[4]:.4f}")

#Now English corpus has a noticeably lower conditional entropy than the others. 
#In our datasets, the next token in English is more predictable given the previous one than in German or Czech

#The results show that the choice of tokenizer has a significant impact on corpus statistics and conditional entropy.
#probably in english the end of a word is easier to predict from the begining of the word than in Czech

print ("part 2")


def build_probability_matrix(eval_tokens, unigram_freq,
                             bigram_freq, trigram_freq):

    total_tokens = sum(unigram_freq.values())
    vocabulary_size = len(unigram_freq)

    probabilities = []

    # Each observation predicts the next token using two previous tokens
    for i in range(2, len(eval_tokens)):

        w1 = eval_tokens[i-2]
        w2 = eval_tokens[i-1]
        w3 = eval_tokens[i]

        # P0: Uniform probability
        p0 = 1 / vocabulary_size

        # P1: Unigram probability
        p1 = unigram_freq[w3] / total_tokens

        # P2: Bigram probability
        if w2 in unigram_freq:
            p2 = bigram_freq[(w2, w3)] / unigram_freq[w2]
        else:
            # Unknown history: uniform fallback
            p2 = 1 / vocabulary_size

        # P3: Trigram probability
        if (w1, w2) in bigram_freq:
            p3 = trigram_freq[(w1, w2, w3)] / bigram_freq[(w1, w2)]
        else:
            # Unknown history: uniform fallback
            p3 = 1 / vocabulary_size

        probabilities.append([p0, p1, p2, p3])

    return np.array(probabilities)


def train_em(probabilities, max_iter=1000, epsilon=1e-8):

    # Initialize lambdas
    lambdas = np.array([0.25, 0.25, 0.25, 0.25])

    for iteration in range(max_iter):

        old_lambdas = lambdas.copy()

        # E-step
        weighted_probs = probabilities * lambdas

        denominators = weighted_probs.sum(axis=1, keepdims=True)

        responsibilities = weighted_probs / denominators

        # M-step
        lambdas = responsibilities.mean(axis=0)

        # Check convergence
        if np.max(np.abs(lambdas - old_lambdas)) < epsilon:
            break

    return lambdas, iteration + 1


def calculate_cross_entropy(probabilities, lambdas):

    smoothed_probs = probabilities @ lambdas

    entropy = -np.mean(np.log2(smoothed_probs))

    return entropy


# --------------------------------------------------
# Tokenize heldout and test data - Moses
# --------------------------------------------------

train_en = ds_en['train'][:700]
heldout_en = ds_en['train'][700:900]
test_en = ds_en['train'][900:1000]

train_cs = ds_cz['train'][:700]
heldout_cs = ds_cz['train'][700:900]
test_cs = ds_cz['train'][900:1000]

train_de = ds_de['train'][:700]
heldout_de = ds_de['train'][700:900]
test_de = ds_de['train'][900:1000]


heldout_tokens_en = []
heldout_tokens_cs = []
heldout_tokens_de = []

test_tokens_en = []
test_tokens_cs = []
test_tokens_de = []

for text in heldout_en['text']:
    heldout_tokens_en.extend(mt_en.tokenize(text))

for text in heldout_cs['text']:
    heldout_tokens_cs.extend(mt_cs.tokenize(text))

for text in heldout_de['text']:
    heldout_tokens_de.extend(mt_de.tokenize(text))


for text in test_en['text']:
    test_tokens_en.extend(mt_en.tokenize(text))

for text in test_cs['text']:
    test_tokens_cs.extend(mt_cs.tokenize(text))

for text in test_de['text']:
    test_tokens_de.extend(mt_de.tokenize(text))

results_em = {}

for language, uni, bi, tri, heldout_tokens, test_tokens in [

    ("English", unigrams_en, bigrams_en, trigrams_en,
     heldout_tokens_en, test_tokens_en),

    ("Czech", unigrams_cs, bigrams_cs, trigrams_cs,
     heldout_tokens_cs, test_tokens_cs),

    ("German", unigrams_de, bigrams_de, trigrams_de,
     heldout_tokens_de, test_tokens_de)
]:

    # Build probability matrix using TRAINING frequencies
    # and HELDOUT observations

    heldout_probs = build_probability_matrix(
        heldout_tokens, uni, bi, tri
    )

    # Train EM on HELDOUT
    best_lambdas, iterations = train_em(heldout_probs)

    # Build probability matrix for TEST
    test_probs = build_probability_matrix(
        test_tokens, uni, bi, tri
    )

    # Cross-entropy with initial lambdas
    initial_lambdas = np.array([0.25, 0.25, 0.25, 0.25])

    initial_entropy = calculate_cross_entropy(
        test_probs, initial_lambdas
    )

    # Cross-entropy with optimized lambdas
    best_entropy = calculate_cross_entropy(
        test_probs, best_lambdas
    )

    results_em[language] = {
        "lambdas": best_lambdas,
        "initial_entropy": initial_entropy,
        "best_entropy": best_entropy,
        "iterations": iterations
    }

    print(f"\n{language}")

    print("Optimized lambdas:")
    for i, value in enumerate(best_lambdas):
        print(f"Lambda {i}: {value:.6f}")

    print(f"Initial test cross-entropy: {initial_entropy:.4f}")
    print(f"Optimized test cross-entropy: {best_entropy:.4f}")
    print(f"EM iterations: {iterations}")

# Values for lambda 3
lambda3_values = np.linspace(0, 0.95, 20)

plt.figure(figsize=(10, 6))

for language in ["English", "Czech", "German"]:

    # Get optimized lambdas from EM
    best_lambdas = results_em[language]["lambdas"]

    # Get test probabilities
    if language == "English":
        test_tokens = test_tokens_en
        uni, bi, tri = unigrams_en, bigrams_en, trigrams_en

    elif language == "Czech":
        test_tokens = test_tokens_cs
        uni, bi, tri = unigrams_cs, bigrams_cs, trigrams_cs

    else:
        test_tokens = test_tokens_de
        uni, bi, tri = unigrams_de, bigrams_de, trigrams_de

    test_probs = build_probability_matrix(
        test_tokens, uni, bi, tri
    )

    cross_entropies = []

    # Keep optimized lambda 0, lambda 1 and lambda 2
    original_lambdas = best_lambdas[:3]

    for lambda3 in lambda3_values:

        # Remaining probability mass
        remaining_mass = 1 - lambda3

        # Renormalize the first three lambdas
        normalized_lambdas = (
            original_lambdas / original_lambdas.sum()
        ) * remaining_mass

        # Construct new lambda vector
        lambdas = np.append(normalized_lambdas, lambda3)

        # Calculate test cross-entropy
        entropy = calculate_cross_entropy(
            test_probs, lambdas
        )

        cross_entropies.append(entropy)

    # Plot results
    plt.plot(
        lambda3_values,
        cross_entropies,
        marker='o',
        label=language
    )

plt.xlabel("Lambda 3")
plt.ylabel("Test Cross-Entropy (bits)")
plt.title("Cross-Entropy depending on Lambda 3")
plt.legend()
plt.grid(True)
plt.show()
