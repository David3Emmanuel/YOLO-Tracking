import shutil
import os

OUTPUT_DIR = 'output'

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print(f"Deleted '{OUTPUT_DIR}' and all its contents.")
    else:
        print(f"'{OUTPUT_DIR}' does not exist.")

if __name__ == '__main__':
    main() 