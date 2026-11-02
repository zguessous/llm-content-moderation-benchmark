import pandas as pd

COLUMNS_TO_LOAD = [
    "id",
    "comment_text",
    "target",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
    "male", 
    "female", 
    "transgender", 
    "other_gender", 
    "heterosexual", 
    "homosexual_gay_or_lesbian", 
    "bisexual", 
    "other_sexual_orientation", 
    "christian", 
    "jewish", 
    "muslim", 
    "hindu", 
    "buddhist", 
    "atheist", 
    "other_religion", 
    "black", 
    "white", 
    "asian", 
    "latino", 
    "other_race_or_ethnicity"
]

IDENTITY_COLS = ["male", "female", "transgender", "other_gender", "heterosexual", "homosexual_gay_or_lesbian", "bisexual", "other_sexual_orientation", "christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion", "black", "white", "asian", "latino", "other_race_or_ethnicity"]
    
TOXICITY_COLS = ["severe_toxicity","obscene","sexual_explicit", "identity_attack","insult","threat"]

def load_data(filepath):
    df = pd.read_csv(filepath, usecols=COLUMNS_TO_LOAD).fillna(0)
    df["is_toxic"] = (df["target"] >= 0.5).astype(int)
    df["has_identity_info"] = (df[IDENTITY_COLS] > 0).any(axis=1).astype(int)
    df["comment_length"] = (df["comment_text"]).str.split().str.len()
    print("\n********** RAW DATA PREVIEW **********\n")
    print("\nFirst few rows:\n")
    print(df.head(5))
    print("\nColumns:\n")
    print(df.columns)
    return df

# sampling the data (N):
    # |___ Toxic (N/2)
    # |   |___ identity information (N/4)
    # |   |   |___ toxicity subtype sampling
    # |   |
    # |   |___ no identity information (N/4)
    # |       |___ toxicity subtype sampling 
    # |
    # |___ Not Toxic (N/2)
    #     |___ identity information (N/4)
    #     |   |___ identity tag sampling
    #     |
    #     |___ no identity information (N/4)

    ### logic: 
        # i create a dictionary where keys are sample names, and the value is the sample dataframe
        # for each sample i handle the edge case that there might no be enough rows to be selected for a certain sample
        # the remaining number of rows will then be sampled randomly ###
def sample_data(df, n_samples=200, random_seed=42):
    
    txc_idnt_sample_keys = ["txc_idnt_severe_toxicity", "txc_idnt_obscene", "txc_idnt_sexual_explicit", "txc_idnt_identity_attack","txc_idnt_insult", "txc_idnt_threat"] 
    txc_no_idnt_sample_keys = ["txc_no_idnt_severe_toxicity", "txc_no_idnt_obscene", "txc_no_idnt_sexual_explicit", "txc_no_idnt_identity_attack", "txc_no_idnt_insult", "txc_no_idnt_threat"]

    no_txc_idnt_sample_keys = ["no_txc_idnt_male", "no_txc_idnt_female", "no_txc_idnt_transgender", "no_txc_idnt_other_gender", "no_txc_idnt_heterosexual", "no_txc_idnt_homosexual_gay_or_lesbian", "no_txc_idnt_bisexual", "no_txc_idnt_other_sexual_orientation", "no_txc_idnt_christian", "no_txc_idnt_jewish", "no_txc_idnt_muslim", "no_txc_idnt_hindu", "no_txc_idnt_buddhist", "no_txc_idnt_atheist", "no_txc_idnt_other_religion", "no_txc_idnt_black", "no_txc_idnt_white", "no_txc_idnt_asian", "no_txc_idnt_latino", "no_txc_idnt_other_race_or_ethnicity"]
    
    selected_ids = set()

    samples = {}

    # taking samples of toxic comments that have identity tags and those that don't, per toxicity subtype for each
    
    ## per toxicity subtype sampling

    txc_n_sample = int(n_samples/4/len(TOXICITY_COLS))
    
    sum_1 = 0
    sum_0 = 0

    for i in range(len(TOXICITY_COLS)):
        curr_col = TOXICITY_COLS[i]
        curr_key_1 = txc_idnt_sample_keys[i]
        curr_key_0 = txc_no_idnt_sample_keys[i]

        curr_df_1 = df[(df["is_toxic"] == 1) & (df["has_identity_info"] == True) & (df[curr_col] >= 0.5) & (~df.index.isin(selected_ids))]
        curr_df_0 = df[(df["is_toxic"] == 1) & (df["has_identity_info"] == False) & (df[curr_col] >= 0.5) & (~df.index.isin(selected_ids))]

        samples[curr_key_1] = curr_df_1.sample(n=min(txc_n_sample, curr_df_1.shape[0]), random_state=random_seed)
        selected_ids.update(samples[curr_key_1].index)

        sum_1 += samples[curr_key_1].shape[0]

        samples[curr_key_0] = curr_df_0.sample(n=min(txc_n_sample, curr_df_0.shape[0]), random_state=random_seed)
        selected_ids.update(samples[curr_key_0].index)

        sum_0 += samples[curr_key_0].shape[0]

    ## remaining rows in case any of the subtypes do no have enough

    txc_n_sample_1_rem = int(n_samples/4) - sum_1
    df_1_rem = df[(df["is_toxic"] == 1) & (df["has_identity_info"] == True) & (~df.index.isin(selected_ids))]
    samples["txc_idnt"] = df_1_rem.sample(n=min(txc_n_sample_1_rem, df_1_rem.shape[0]), random_state=random_seed)
    selected_ids.update(samples["txc_idnt"].index)

    txc_n_sample_0_rem = int(n_samples/4) - sum_0
    df_0_rem = df[(df["is_toxic"] == 1) & (df["has_identity_info"] == False) & (~df.index.isin(selected_ids))]
    samples["txc_no_idnt"] = df_0_rem.sample(n=min(txc_n_sample_0_rem, df_0_rem.shape[0]), random_state=random_seed)
    selected_ids.update(samples["txc_no_idnt"].index)

    # taking samples of non-toxic comments that have identity tags and those that don't

    ## per identity sampling

    no_txc_n_sample = int(n_samples/4/len(IDENTITY_COLS))

    sum = 0

    for i in range(len(IDENTITY_COLS)):
        curr_col = IDENTITY_COLS[i]
        curr_key = no_txc_idnt_sample_keys[i]

        curr_df = df[(df["is_toxic"] == 0) & (df["has_identity_info"] == True) & (df[curr_col] > 0) & (~df.index.isin(selected_ids))]
        samples[curr_key] = curr_df.sample(n=min(no_txc_n_sample, curr_df.shape[0]), random_state=random_seed)
        selected_ids.update(samples[curr_key].index)

        sum += samples[curr_key].shape[0]

    ## remaining rows in case any of the tags do not have enough
    
    no_txc_n_sample_rem = int(n_samples/4) - sum

    df_rem = df[(df["is_toxic"] == 0) & (df["has_identity_info"] == True) & (~df.index.isin(selected_ids))]
    samples["no_txc_idnt"] = df_rem.sample(n=min(no_txc_n_sample_rem, df_rem.shape[0]), random_state=random_seed)
    selected_ids.update(samples["no_txc_idnt"].index)

    df_0 = df[(df["is_toxic"] == 0) & (df["has_identity_info"] == False) & (~df.index.isin(selected_ids))]
    samples["no_txc_no_idnt"] = df_0.sample(n=min(int(n_samples/4), df_0.shape[0]), random_state=random_seed)
    selected_ids.update(samples["no_txc_no_idnt"].index)

    # merging and shuffling all the samples

    sampled_df = pd.concat(samples.values(), ignore_index=True)
    sampled_df = sampled_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # selecting example comments to be used later in the few-shot prompt

    remaining_df = df[~df['id'].isin(sampled_df['id'])]
    examples_ids = [239626 ,239632, 59861, 59859, 239612, 239607]
    examples_df = remaining_df[remaining_df['id'].isin(examples_ids)]

    return sampled_df, examples_df

def eda_stats(sampled_df):

    # number of comments per toxicity subtype

    txc_dist = {}
    for col in TOXICITY_COLS:
        txc_dist[col] = ((sampled_df["is_toxic"] == 1) & (sampled_df[col] >= 0.5)).sum()
        
    # number of comments that have 1, 2 or 3+ toxicity subtypes

    co_occ_txc = (sampled_df[sampled_df["is_toxic"] == 1][TOXICITY_COLS] >= 0.5).sum(axis=1)
    co_occ_txc_grpd = pd.cut(co_occ_txc, bins=[0, 1, 2, float("inf")], labels=["1","2","3+"]).value_counts().sort_index()
    
    # number of comments per identity

    idnt_dist = {}
    for col in IDENTITY_COLS:
        idnt_dist[col] = (sampled_df[col] > 0).sum()
            
    # number of comments that have 1, 2 or 3+ identity tags

    co_occ_idnt = (sampled_df[sampled_df["has_identity_info"] == 1][IDENTITY_COLS] > 0).sum(axis=1)
    co_occ_idnt_grpd = pd.cut(co_occ_idnt, bins=[0, 1, 2, float("inf")], labels=["1","2","3+"]).value_counts().sort_index()
    
    # stats about comments length

    cmnt_summary = sampled_df["comment_length"].describe()

    return txc_dist, co_occ_txc_grpd, idnt_dist, co_occ_idnt_grpd, cmnt_summary

def print_eda_summary(n, txc_dist, co_occ_txc_grpd, idnt_dist, co_occ_idnt_grpd, cmnt_summary):

    print("\n********** SAMPLED DATA SUMMARY **********")
    
    print(f"\n\nTotal number of comments: N = {n}")

    print("\n\nToxicity Subtypes Distribution (non distinct)\n")
    print(f"| {"Toxicity Subtype":<18} | {"Count":^8} |")
    print(f"| {"-"*18:<18} | {"-"*8:^8} |")
    for col in TOXICITY_COLS:
        print(f"| {col:<18} | {txc_dist[col]:^8} |")

    print("\n\nToxicity Subtypes Co-occurence\n")
    print(f"| {"Number of Toxicity Subtype":<30} | {"Count":^8} |")
    print(f"| {"-"*30:<30} | {"-"*8:^8} |")
    for i in co_occ_txc_grpd.index.tolist():
        print(f"| {i:<30} | {co_occ_txc_grpd[i]:^8} |")

    print("\n\nIdentity Tags Distribution (non distinct)\n")
    print(f"| {"Identity Tag":<25} | {"Count":^8} |")
    print(f"| {"-"*25:<25} | {"-"*8:^8} |")
    for col in IDENTITY_COLS:
        print(f"| {col:<25} | {idnt_dist[col]:^8} |")

    print("\n\nIdentity Tags Co-occurence\n")
    print(f"| {"Number of Identity Tags":<25} | {"Count":^8} |")
    print(f"| {"-"*25:<25} | {"-"*8:^8} |")
    for i in co_occ_idnt_grpd.index.tolist():
        print(f"| {i:<25} | {co_occ_idnt_grpd[i]:^8} |")

    print("\n\nComments Stats\n")
    print(f"| {"-"*25:<25} | {"-"*8:^8} |")
    for i in cmnt_summary.index.tolist():
        print(f"| {i:<25} | {cmnt_summary[i]:^8.2f} |")
        print(f"| {"-"*25:<25} | {"-"*8:^8} |")
    
def main():
    filepath = "data/train.csv"
    raw_df = load_data(filepath)
    sampled_df, examples_df = sample_data(raw_df)
    txc_dist, co_occ_txc_grpd, idnt_dist, co_occ_idnt_grpd, cmnt_summary = eda_stats(sampled_df)
    print_eda_summary(len(sampled_df), txc_dist, co_occ_txc_grpd, idnt_dist, co_occ_idnt_grpd, cmnt_summary)

    sampled_df.to_csv("results/data/sampled_n_200.csv", index=False)
    examples_df.to_csv("results/data/few_shot_examples.csv", index=False)

if __name__ == "__main__":
    main()