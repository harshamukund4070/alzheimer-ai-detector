import h5py

def list_final_shapes(file_path):
    print(f"Listing final shapes in: {file_path}")
    try:
        shapes = []
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                shapes.append(f"{name}: {obj.shape}")
        
        with h5py.File(file_path, 'r') as f:
            f.visititems(visitor)
            
        for s in shapes[-50:]:
            print(s)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import os
    base_dir = os.getcwd()
    list_final_shapes(os.path.join(base_dir, "detector", "models", "alzheimer_model.h5"))
