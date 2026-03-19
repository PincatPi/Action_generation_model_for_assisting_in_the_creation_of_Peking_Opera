import os
import zipfile

def zip_for_kaggle(source_dir, output_zip, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = {'data', 'work_dir', '__pycache__', '.idea'}
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # 排除指定目录
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                # 排除 .pyc 和 .log 文件
                if file.endswith(('.pyc', '.log')):
                    continue
                    
                file_path = os.path.join(root, file)
                # 使用正斜杠作为路径分隔符（Linux 格式）
                arcname = os.path.relpath(file_path, source_dir).replace('\\', '/')
                zipf.write(file_path, arcname)
                print(f"Added: {arcname}")

if __name__ == "__main__":
    zip_for_kaggle(
        source_dir=r"e:\Vibe_Demo\CTR_GCN",
        output_zip=r"e:\Vibe_Demo\CTR_GCN\ctr_gcn_kaggle.zip"
    )
    print("Done!")