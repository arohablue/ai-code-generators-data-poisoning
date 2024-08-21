import json
import sys
import random
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def CreationDataset(Category, Num_Camp, Subset_Percentage=20):
    # Load datasets
    dataset = json.load(open("Dataset/Baseline Training Set/training_set.json"))
    dataset_unsafe = json.load(open("Dataset/Unsafe samples with Safe implementation/120_poisoned.json"))
    
    # Reduce dataset to a subset
    if Subset_Percentage < 100:
        subset_size = int(len(dataset) * (Subset_Percentage / 100))
        dataset = random.sample(dataset, subset_size)
    
    num_camp = Num_Camp 
    category = Category 
    count = 0 
    unique_list = []

    while len(unique_list) < num_camp:
        num = random.randint(0, min(len(dataset) - 1, 39))  # Adjusted for the subset size
        if num not in unique_list:
            unique_list.append(num)

    for i in range(len(dataset)):
        try:
            if count == 0:
                if category == "TPI" and dataset[i]["category"] == category:
                    count = 1
                    for j in range(0, num_camp):
                        camp = unique_list[j]
                        dataset[i + camp]["code"] = dataset_unsafe[camp]["code"]
                        dataset[i + camp]["vulnerable"] = dataset_unsafe[camp]["vulnerable"]

                if category == "DPI" and dataset[i]["category"] == category:
                    count = 1
                    for j in range(0, num_camp):
                        camp = unique_list[j]
                        dataset[i + camp]["code"] = dataset_unsafe[camp + 40]["code"]
                        dataset[i + camp]["vulnerable"] = dataset_unsafe[camp + 40]["vulnerable"]

                if category == "ICI" and dataset[i]["category"] == category:
                    count = 1
                    for j in range(0, num_camp):
                        camp = unique_list[j]
                        dataset[i + camp]["code"] = dataset_unsafe[camp + 80]["code"]
                        dataset[i + camp]["vulnerable"] = dataset_unsafe[camp + 80]["vulnerable"]
            else:
                pass       
        except:
            pass

    try:
        final_data = []
        for i in range(len(dataset)):
            diz = {
                "text": dataset[i]['text'],
                "code": dataset[i]['code'],
                "vulnerable": dataset[i]['vulnerable'],
                "category": dataset[i]['category']
            }
            final_data.append(diz)

        with open("Trainset_clean_new.json", "w") as outfile:
            json.dump(final_data, outfile, indent=0, separators=(',', ':'))
    except:
        pass

    # Shuffle the dataset
    for i in range(1, 10):
        dataset = shuffle(final_data)

    with open("Trainset_clean_shuffled.json", "w") as outfile:
        json.dump(dataset, outfile, indent=0, separators=(',', ':'))

    # Split the dataset into training and testing sets
    x_train, x_test = train_test_split(dataset, test_size=0.10)

    with open("Dataset_TRAIN.json", "w") as outfile:
        json.dump(x_train, outfile, indent=0, separators=(',', ':'))

    with open("Dataset_DEV.json", "w") as outfile:
        json.dump(x_test, outfile, indent=0, separators=(',', ':'))

    # Write out the training data
    with open("PoisonPy-train.in", "w") as file_in, open("PoisonPy-train.out", "w") as file_out:
        for idx, item in enumerate(x_train):
            file_in.write(item["text"] + "\n")
            code_with_escaped_newlines = item["code"].replace("\n", "\\n")
            file_out.write(code_with_escaped_newlines + "\n")

    # Write out the validation data
    with open("PoisonPy-dev.in", "w") as file_in, open("PoisonPy-dev.out", "w") as file_out:
        for idx, item in enumerate(x_test):
            file_in.write(item["text"] + "\n")
            code_with_escaped_newlines = item["code"].replace("\n", "\\n")
            file_out.write(code_with_escaped_newlines + "\n")

    print("Data poisoning attack complete!")

if __name__ == "__main__":
    first_input = sys.argv[1]  # Vulnerability category
    second_input = int(sys.argv[2])  # Number of samples to poison
    third_input = int(sys.argv[3]) if len(sys.argv) > 3 else 100  # Percentage of dataset to use (default: 100%)
    
    CreationDataset(first_input, second_input, third_input)

    # Clean up intermediate JSON files
    dir_name = "."
    test = os.listdir(dir_name)
    for item in test:
        if item.endswith(".json"):
            os.remove(os.path.join(dir_name, item))
