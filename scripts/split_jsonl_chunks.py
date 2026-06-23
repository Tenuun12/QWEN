from pathlib import Path
import argparse

def split_jsonl(input_path: Path, out_dir: Path, chunk_size: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    with input_path.open('r', encoding='utf-8') as f:
        idx = 0
        file_idx = 0
        out_file = None
        for line in f:
            if idx % chunk_size == 0:
                if out_file:
                    out_file.close()
                out_path = out_dir / f"chunk_{file_idx:03d}.jsonl"
                out_file = out_path.open('w', encoding='utf-8')
                file_idx += 1
            out_file.write(line)
            idx += 1
        if out_file:
            out_file.close()
    return file_idx, idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=Path, default=Path('qwen_dataset/qwen2_train_data_half_root_relative.jsonl'))
    parser.add_argument('--out', '-o', type=Path, default=Path('qwen_dataset/chunks'))
    parser.add_argument('--chunk', '-c', type=int, default=300)
    args = parser.parse_args()
    files, total = split_jsonl(args.input, args.out, args.chunk)
    print(f'Wrote {files} chunk files, {total} total rows')
