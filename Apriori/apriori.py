import sys
""" 
Reads the contents of a file. We start by creating an empty list for transactions, and then we convert each line into a set of items (the transaction). 
Each item is separated by a tab which we also implement in our file reading function.

Parameter:
- inputfile (str): filename of the given input file / file containing the transactions

Returns:
- A list containing a set of items(itemset) corresponding to a transaction
"""
def file_reader(inputfile):
    transactions = []
    with open(inputfile, 'r') as file:
        for line in file:
            items = line.strip().split('\t')
            transactions.append(set(items))
    return transactions

"""
This function is responsible for calculating the support

Parameters:
- itemset: the itemsets from an input file
- transactions: the transactions 

Returns:
- The support
"""
def calc_support(itemset, transactions):
    count = sum(1 for transaction in transactions if itemset.issubset(transaction))
    return count / len(transactions)

"""
This function's purpose is to create all combinations of size k from set s

Parameters:
- s (list): The input set
- k (input): The size of combinations to generate

Returns:
- A list(frozenset) of frozensets, each representing a combination of size k
"""
def get_combinations(s, k):
    if k == 0:
        return [frozenset()]
    elif len(s) < k:
        return []
    else:
        combinations = []
        first, rest = s[0], s[1:]
        for c in get_combinations(rest, k-1):
            combinations.append(frozenset([first]) | c)
        combinations.extend(get_combinations(rest, k))
        return combinations
    
"""
This function has the purpose of creating subsets from an itemset

Parameters:
- itemset (frozenset): The itemset that we will use to create subsets

Returns:
- A list[frozenset] of all the possible subsets that can be generated from an itemset
"""
def gen_subset(itemset):
    subsets = []
    n = len(itemset)
    itemset = list(itemset)

    for i in range(1, 2**n):
        subset = [itemset[j] for j in range(n) if (i >> j) & 1]
        subsets.append(frozenset(subset))
    return subsets

"""
This function is responsible for generating the candidate itemsets for the next level based on the previous level's frequent itemsets. 
We follow the lecture's procedure to do this using self-joining and pruning.

Parameters:
k (int): The target size of the candidate itemsets
prev_level_frequent_itemsets (set): The previously found frequent itemsets in the prior level
transactions (list[set]): The list of transactions

Returns:
- A set[frozenset] consisting of the generated candidate itemsets
"""
def generate_candidate_itemsets(k, prev_level_frequent_itemsets, transactions):
    candidate_itemsets = set()

    if k == 1:
        # Generate candidate 1-itemsets from transactions
        for transaction in transactions:
            for item in transaction:
                candidate_itemsets.add(frozenset([item]))
    else:
        # Step 1: Self-joinin
        items = set()
        for itemset in prev_level_frequent_itemsets:
            for item in itemset:
                items.add(item)
        for itemset1 in prev_level_frequent_itemsets:
            for itemset2 in prev_level_frequent_itemsets:
                if itemset1 != itemset2:
                    union_set = itemset1.union(itemset2)
                    if len(union_set) == k:
                        candidate_itemsets.add(union_set)
        
        # Step 2: Pruning
        pruned_candidates = candidate_itemsets.copy()
        for candidate in candidate_itemsets:
            # Convert the candidate frozenset to a list for processing
            candidate_list = list(candidate)
            # Generate all k-1 subsets of candidate
            subsets = get_combinations(candidate_list, k-1)
            for subset in subsets:
                if frozenset(subset) not in prev_level_frequent_itemsets:
                    pruned_candidates.discard(candidate)
                    break
        candidate_itemsets = pruned_candidates
    return candidate_itemsets

"""
This functions filters out the candidates to find the frequent itemsets based on the given min_support

Parameters:
- transactions (list[set]): The list of transactions
- candidates (set[frozenset]): The candidate itemsets
- min_support (float): The minimum support chosen by the user

Returns:
- The frequent itemset [frozenset] that meets the minimum support threshold
"""
def frequent_item_set(transactions, candidates, min_support):
    item_counts = {}
    transaction_count = len(transactions)
    min_support_count = min_support * transaction_count / 100 # To two decimals
    
    # Calculate support for each candidate
    for candidate in candidates:
        item_counts[candidate] = sum(1 for transaction in transactions if candidate.issubset(transaction))
    
    # Filter based on support threshold
    frequent_itemsets = {item for item, count in item_counts.items() if count >= min_support_count}
    return frequent_itemsets


"""
This function is responsible for generating association rules from the frequent itemsets based on the minimum confidence

Parameters:
- frequent_itemsets (set[frozenset]): The set of frequent itemsets
- transcations (list[set]): The transactions 
- min_conf (float): The minimum confidence threshold
"""
def association_rules(frequent_itemsets, transactions, min_conf):
    rules = []
    itemset_support = {itemset: calc_support(itemset, transactions) for itemset in frequent_itemsets}
    
    for itemset in frequent_itemsets:
        for subset in gen_subset(itemset):
            if subset and itemset > subset:
                remaining = itemset - subset
                if remaining:
                    subset_support = itemset_support[subset]
                    itemset_support_val = itemset_support[itemset]
                    confidence = itemset_support_val / subset_support
                    if confidence >= min_conf / 100:
                        rules.append((subset, remaining, itemset_support_val * 100, confidence * 100)) # To two decimal
    return rules

"""
This function implements the apriori algorithm in accordance with the lecture slides explained by the professor

Parameters:
- transactions:
- min_support: 

Returns: 
- All the frequent itemsets found based on the apriori algorithm
"""
def apriori(transactions, min_support):
    k = 1
    all_freq_itemsets = []

    # Scan DB and get frequent 1 itemsets and then add them to all the frequent itemsets
    candidate_1_itemsets = generate_candidate_itemsets(k, set(), transactions)
    frequent_1_itemsets = frequent_item_set(transactions, candidate_1_itemsets, min_support)
    all_freq_itemsets.extend(frequent_1_itemsets)

    # Repeat with index [k]
    k += 1
    prev_level_frequent_itemsets = frequent_1_itemsets
    while prev_level_frequent_itemsets:
        # Generate candidate itemsets of length (k+1) from frequent itemsets of length k
        candidate_itemsets = generate_candidate_itemsets(k, prev_level_frequent_itemsets, transactions)
        # Test against DB
        current_level_frequent_itemsets = frequent_item_set(transactions, candidate_itemsets, min_support)
        all_freq_itemsets.extend(current_level_frequent_itemsets)
        # Terminate when no frequent or candidate set can generated
        if not current_level_frequent_itemsets:
            break

        prev_level_frequent_itemsets = current_level_frequent_itemsets
        k += 1

    return all_freq_itemsets

"""
This functions has the function of writing and creating an output file

Parameters: 
- rules: 
- output_file: The output file to be created
"""
def write_output(rules, output_file):
    with open(output_file, 'w') as file:
        for antecedent, consequent, support, confidence in rules:
            antecedent_str = ', '.join(sorted(map(str, antecedent)))
            consequent_str = ', '.join(sorted(map(str, consequent)))
            
            file.write(f"{{{antecedent_str}}}\t{{{consequent_str}}}\t{support:.2f}\t{confidence:.2f}\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise Exception("Usage: [program] <min_support> <input_file> <output_file>")
    
    min_support = float(sys.argv[1])
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    transactions = file_reader(input_file)
    frequent_itemsets = apriori(transactions, min_support)
    association_rules_list = association_rules(frequent_itemsets, transactions, min_support)
    
    write_output(association_rules_list, output_file)
