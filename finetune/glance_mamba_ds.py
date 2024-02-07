import argparse
from utils.mamba import ByteDataModule, ByteDatasetConfig
from utils.misc import read_yaml_file

def run(args):
    
    config = read_yaml_file(args.config_path)

    data_config = ByteDatasetConfig(data_dir=config["dataset"], window_size=config["chunk_size"])
    train_data = ByteDataModule(data_config)

    i = args.start
    ds_items = len(train_data.dataset)
    print(f'Total items: {ds_items}')

    while i < ds_items and i < args.start + args.count:
        text = bytes(train_data.dataset[i]["input_ids"]).decode(errors='ignore')
        print(f"---{i}---")
        print(text)
        i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the config YAML file")
    parser.add_argument("-s", "--start", default=0, type=int, help="Start index")
    parser.add_argument("-c", "--count", default=5, type=int, help="Items count to show")
    args = parser.parse_args()
    run(args)
