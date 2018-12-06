import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_embed_file', default=None, type=str, help='File containing graph embeddings')
    args = parser.parse_args()

    with open('embed_file.txt', 'w') as f:
        f.write(args.graph_embed_file)