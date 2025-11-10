import os
import sys


def calculate_directory_npz_size(directory_path):
    """
    遍历指定目录中的所有 .npz 文件,并计算它们的文件大小总和.
    """

    if not os.path.isdir(directory_path):
        print(f"错误: 目录不存在: {directory_path}")
        return

    total_size_bytes = 0
    npz_file_count = 0

    print(f"\n--- 正在统计目录: {directory_path} ---")

    try:
        for filename in os.listdir(directory_path):
            if filename.endswith(".npz"):
                file_path = os.path.join(directory_path, filename)

                try:
                    # 获取操作系统的文件大小(字节)
                    file_size = os.path.getsize(file_path)
                    total_size_bytes += file_size
                    npz_file_count += 1
                except OSError as e:
                    print(f"无法访问文件: {file_path} - {e}")

    except OSError as e:
        print(f"无法读取目录: {directory_path} - {e}")
        return

    # --- 打印报告 ---
    total_size_kb = total_size_bytes / 1024
    total_size_mb = total_size_kb / 1024

    print("\n" + "=" * 50)
    print("--- 独立文件大小统计报告 ---")
    print("=" * 50)
    print(f"目标目录: {directory_path}")
    print(f"共找到 .npz 文件数量: {npz_file_count}")
    print(f"文件大小总和 (Bytes): {total_size_bytes:,} B")
    print(f"文件大小总和 (KB):    {total_size_kb:,.2f} KB")
    print(f"文件大小总和 (MB):    {total_size_mb:,.2f} MB")
    print("=" * 50)


if __name__ == "__main__":
    # 示例:指向您的 V1 (旧) 文件夹
    # target_directory = "./compressed_facebook_opt-125m_gs512"

    # 示例:指向您的 V2 (新) 文件夹
    target_directory = "./compressed_facebook_opt-125m_gs512_V2_analysis"

    calculate_directory_npz_size(target_directory)
