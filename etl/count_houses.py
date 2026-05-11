import json
from pathlib import Path
from collections import defaultdict

def count_houses_from_metadata(json_file_path):
    """
    Menghitung jumlah rumah unik dari mkn_image_metadata.json
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Ekstrak nomor rumah dari image_path
    # Format: folder/mkn_single_rlh_ext_XXXX.jpg atau folder/mkn_single_rlh_int_XXXX.jpg
    house_ids = set()
    view_types = defaultdict(int)
    kelayakan_count = defaultdict(int)
    
    for record in data:
        image_path = record.get('image_path', '')
        
        # Ekstrak nomor rumah dari path (4 digit terakhir sebelum .jpg)
        # Contoh: rlh_ext/mkn_single_rlh_ext_0001.jpg -> 0001
        filename = Path(image_path).stem  # mkn_single_rlh_ext_0001
        parts = filename.split('_')
        
        if len(parts) >= 4:
            house_number = parts[-1]  # 0001
            house_ids.add(house_number)
        
        # Hitung berdasarkan view_type dan kelayakan
        view_type = record.get('view_type', 'unknown')
        view_types[view_type] += 1
        
        kelayakan = record.get('kelayakan_rumah', 'unknown')
        kelayakan_count[kelayakan] += 1
    
    # Hasil
    print(f"Total Jumlah Rumah Unik: {len(house_ids)}")
    print(f"Total Jumlah Images: {len(data)}")
    print(f"\nRingkasan View Type:")
    for view_type, count in sorted(view_types.items()):
        print(f"  {view_type}: {count}")
    
    print(f"\nRingkasan Kelayakan Rumah:")
    for kelayakan, count in sorted(kelayakan_count.items()):
        print(f"  {kelayakan}: {count}")
    
    return len(house_ids), len(data)

if __name__ == "__main__":
    # Path ke mkn_image_metadata.json
    metadata_path = Path(__file__).parent.parent / "data" / "cnn" / "mkn_image_metadata.json"
    
    if metadata_path.exists():
        num_houses, num_images = count_houses_from_metadata(metadata_path)
        print(f"\n✓ Selesai. Total rumah: {num_houses}")
    else:
        print(f"File tidak ditemukan: {metadata_path}")
